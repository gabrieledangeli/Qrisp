# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Parameter-shift Jacobians and derivatives for QML circuits."""

from collections.abc import Callable
from math import pi

import jax.numpy as jnp
import numpy as np
from jax import Array, config, core, custom_vjp
from jax._src.interpreters.ad import LinearizeTrace
from jax._src.interpreters.batching import BatchTrace
from jax.core import Tracer
from jax.custom_batching import custom_vmap
from jax.lax import dynamic_update_slice

from qrisp import QuantumVariable, reset
from qrisp.algorithms.qml._primitives import assert_condition_p
from qrisp.algorithms.qml.execution import expectation_pauli
from qrisp.jasp import (
    AbstractQubit,
    AbstractQubitArray,
    Jaspr,
    QuantumPrimitive,
    TracingModeError,
    TracingQuantumSession,
    collect_environments,
    get_last_equation,
    q_fori_loop,
    qache,
)
from qrisp.jasp.interpreter_tools.call_graph_analysis import _get_jaxpr_core, _iter_sub_jaxprs
from qrisp.operators.qubit import X, Y, Z
from qrisp.typing import NDArrayLike

_PARAMETER_SHIFT = pi / 2


def parameter_shift_jacobian(
    state_preparation: Callable,
    angles: np.ndarray | Array,
    *,
    observable: Callable | tuple[Callable, ...] = Z,
    qubit_index: int | tuple[int, ...] = 0,
) -> dict:
    r"""Compute the Jacobian of local Pauli expectations by shifting each angle by +pi/2 and -pi/2.

    Call inside Jasp with ``jax_enable_x64=True``. Each derivative is half the
    difference between the two shifted expectations.

    Parameters
    ----------
    state_preparation : callable
        Function of the angles that allocates one register outside loops and
        branches and returns it whole. Keep the register size, qubit order and
        circuit structure fixed. Use each angle directly in exactly one RX, RY
        or RZ gate. All other gates must be unitary and independent of the angles.
    angles : numpy.ndarray or jax.Array
        Non-empty 1D array of finite float32 or float64 angles in radians.
        Both shifts must be accurate to within 1e-7 radians.
    observable : callable or tuple of callables, optional
        Pauli observable or non-empty tuple of observables, as in
        :func:`~qrisp.algorithms.qml.execution.expectation_pauli`.
        Choose them at tracing time. The default is ``Z``.
    qubit_index : int or tuple of int, optional
        Qubit index or non-empty tuple of indices, known at tracing time. The default is 0.

    Returns
    -------
    dict
        ``values`` is a 3D float64 array with one derivative per selected qubit,
        observable and angle. The tuples ``qubit_indices``, ``observables`` and
        ``parameter_indices`` label the axes in that order.

    Examples
    --------
    >>> from qrisp.algorithms.qml.gradients import parameter_shift_jacobian
    >>> from qrisp.algorithms.qml.templates import prepare_rotation
    >>> from qrisp.jasp import jaspify
    >>> @jaspify
    ... def z_jacobian(angles):
    ...     return parameter_shift_jacobian(prepare_rotation, angles)["values"]

    """
    return _parameter_shift_jasp(state_preparation, angles, observable, qubit_index)


def _parameter_shift_jasp(state_preparation, angles, observable, qubit_index, *, evaluation="jacobian"):
    """Evaluate the requested unshifted readout and shifted pairs in one Jasp loop."""
    qs = TracingQuantumSession.get_instance()

    if qs.abs_qst is None:
        raise TracingModeError("Use parameter_shift_jacobian inside jaspify or qrisp.jasp.qjit")

    qs._check_in_scope()
    if not callable(state_preparation):
        raise TypeError("state_preparation must be callable")

    angles = _validate_shift_angles(angles, qs)
    observables, indices = _normalize_shift_selection(observable, qubit_index)
    num_angles = angles.shape[0]

    # Reserve the first readout for the original angles when the caller needs the circuit value.
    first_shift, num_evaluations = {
        "jacobian": (0, 2 * num_angles),
        "value": (1, 1),
        "value_and_jacobian": (1, 1 + 2 * num_angles),
    }[evaluation]

    @qache
    def prepare(shifted_angles):
        qv = state_preparation(shifted_angles)

        if not isinstance(qv, QuantumVariable):
            raise TypeError("state_preparation must return a QuantumVariable")

        single_register = len(qs.qv_list) == 1 and qs.qv_list[0] is qv

        if qv.qs is not qs or not single_register or qs.deleted_qv_list:
            raise ValueError("state_preparation must allocate exactly one fresh register")

        return qv

    def evaluate_shift(step, carry):
        readouts, register_size = carry

        # Each pair of steps shifts one angle by +pi/2 and then -pi/2.
        # For the optional leading readout, angle_index is -1 and all offsets are zero.
        shift_index = step - first_shift
        angle_index = shift_index // 2

        shift = jnp.where(shift_index % 2 == 0, _PARAMETER_SHIFT, -_PARAMETER_SHIFT)
        offsets = jnp.where(jnp.arange(num_angles) == angle_index, shift, 0.0)

        qv = prepare(angles + offsets)

        # The last equation holds the circuit traced by the call to prepare.
        _validate_shift_preparation(get_last_equation())

        size = qv.size
        qs.abs_qst = assert_condition_p.bind(
            (step == 0) | (size == register_size),
            qs.abs_qst,
            message="state_preparation must use a constant register size across parameter shifts",
        )

        values = expectation_pauli(qv, observable=observables, qubit_index=indices)["values"]

        # The readout leaves the qubits in their prepared state. Reset them before releasing them.
        reset(qv)
        qv.delete()

        return dynamic_update_slice(readouts, values[..., None], (0, 0, step)), size

    shape = (len(indices), len(observables), num_evaluations)
    initial = (jnp.zeros(shape, dtype=jnp.float64), jnp.int64(-1))

    readouts, _ = q_fori_loop(0, num_evaluations, evaluate_shift, initial)

    plus = readouts[..., first_shift::2]
    minus = readouts[..., first_shift + 1 :: 2]

    result = {
        "values": (plus - minus) / 2,
        "qubit_indices": indices,
        "observables": tuple(pauli.__name__ for pauli in observables),
        "parameter_indices": tuple(range(num_angles)),
    }

    if first_shift:
        result["readout"] = readouts[..., 0]

    return result


def _validate_shift_angles(angles, qs):
    """Return float64 angles whose positive and negative shifts are representable."""
    if not isinstance(angles, NDArrayLike) or angles.dtype not in (np.float32, np.float64):
        raise TypeError("Jasp angles must be a one-dimensional array with dtype float32 or float64")

    if angles.ndim != 1 or not angles.shape[0]:
        raise ValueError("angles must be a non-empty one-dimensional array")

    if not config.x64_enabled:
        raise ValueError("Parameter shifts in Jasp require jax_enable_x64=True")

    angles = jnp.asarray(angles, dtype=jnp.float64)

    # For large angles, rounding can change the size of the shift.
    plus_delta = (angles + _PARAMETER_SHIFT) - angles
    minus_delta = angles - (angles - _PARAMETER_SHIFT)
    deltas = jnp.stack((plus_delta, minus_delta))

    valid = jnp.all(jnp.isfinite(angles)) & jnp.allclose(deltas, _PARAMETER_SHIFT, atol=1e-7, rtol=0)

    qs.abs_qst = assert_condition_p.bind(
        valid, qs.abs_qst, message="angles must be finite and resolve pi/2 shifts to within 1e-7 radians"
    )
    return angles


def _normalize_shift_selection(observable, qubit_index):
    """Collect the static Pauli constructors and indices in the requested order."""
    observables = observable if isinstance(observable, tuple) else (observable,)
    if not observables:
        raise ValueError("observable must contain at least one Pauli constructor")
    for pauli in observables:
        if not any(pauli is constructor for constructor in (X, Y, Z)):
            raise TypeError("observable must be X, Y, Z or a tuple of these Qrisp constructors")

    if qubit_index is None:
        raise TracingModeError("Parameter shifts in Jasp require explicit qubit indices")
    indices = qubit_index if isinstance(qubit_index, tuple) else (qubit_index,)
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("qubit_index must contain Python or NumPy integer scalars; booleans are not supported")
    if not indices or any(index < 0 for index in indices):
        raise ValueError("qubit_index must select at least one non-negative index")
    return observables, tuple(int(index) for index in indices)


def _validate_shift_preparation(eqn):
    """Require one unconditional register allocation and unitary preparation."""
    jaspr = Jaspr.from_cache(collect_environments(eqn.params["jaxpr"])).flatten_environments()
    inputs = (*jaspr.invars, *jaspr.constvars)
    if any(isinstance(var.aval, (AbstractQubit, AbstractQubitArray)) for var in inputs):
        raise ValueError("state_preparation must not capture external quantum registers")

    allowed_primitives = {
        "jasp.get_qubit",
        "jasp.get_size",
        "jasp.slice",
        "jasp.fuse",
        "jasp.assert_condition",
    }
    # Track whether each nested call is inside a loop or branch, where allocation is forbidden.
    pending = [(jaspr, False)]
    allocation_count = 0
    gates = []
    while pending:
        current, in_control_flow = pending.pop()
        for instruction in _get_jaxpr_core(current).eqns:
            primitive = instruction.primitive
            name = primitive.name
            if name == "jasp.create_qubits":
                allocation_count += 1
                if in_control_flow:
                    raise ValueError("state_preparation must allocate its register outside loops and branches")
            elif name == "jasp.quantum_gate":
                gates.append(instruction.params["gate"])
            elif isinstance(primitive, QuantumPrimitive) and name not in allowed_primitives:
                raise ValueError("state_preparation must contain only unitary quantum operations")

            nested_control_flow = in_control_flow or name in ("while", "cond", "scan")
            for sub_jaxpr in _iter_sub_jaxprs(instruction):
                pending.append((sub_jaxpr, nested_control_flow))

    if allocation_count != 1:
        raise ValueError("state_preparation must allocate exactly one fresh register")
    _validate_shift_register(jaspr)

    # A composite gate can contain a measurement or reset in its definition.
    while gates:
        gate = gates.pop()
        if gate.name in ("measure", "reset", "qb_dealloc") or gate.num_clbits:
            raise ValueError("state_preparation must contain only unitary quantum operations")
        if gate.definition is not None:
            gates.extend(instruction.op for instruction in gate.definition.data)
    eqn.params["jaxpr"] = jaspr


def _validate_shift_register(jaspr):
    """Verify that the returned register is the complete original allocation."""
    jaxpr = _get_jaxpr_core(jaspr)
    registers = [var for var in jaxpr.outvars if isinstance(var.aval, AbstractQubitArray)]
    if len(registers) != 1:
        raise ValueError("state_preparation must return exactly one complete register")

    register = registers[0]
    callers = []
    while True:
        if register in jaxpr.invars:
            # Follow a callee's input back to the argument supplied by its caller.
            if not callers:
                break
            position = jaxpr.invars.index(register)
            jaxpr, call = callers.pop()
            register = call.invars[position]
            continue

        producer = next((eqn for eqn in jaxpr.eqns if register in eqn.outvars), None)
        if producer is None:
            break
        if producer.primitive.name == "jasp.create_qubits" and register is producer.outvars[0]:
            return
        # A slice or a join can change which qubits the returned register contains.
        if producer.primitive.name not in ("jit", "pjit"):
            break

        # Follow the matching output into the nested call.
        position = producer.outvars.index(register)
        callers.append((jaxpr, producer))
        jaxpr = _get_jaxpr_core(producer.params["jaxpr"])
        register = jaxpr.outvars[position]
    raise ValueError("state_preparation must return the complete allocated register without replacing its qubit array")


def _batch_expectation(evaluate, angles, state, readout_shape, *, with_jacobian):
    """Lower a mapped QML evaluation to a loop carrying an unbatched state."""
    qs = TracingQuantumSession.get_instance()

    def run_with_state(state, function, *args, **kwargs):
        depth = len(qs.abs_qst_stack)
        qs.start_tracing(state)
        try:
            result = function(*args, **kwargs)
            return result, qs.abs_qst
        finally:
            # A failed loop body may leave additional tracing scopes open.
            while len(qs.abs_qst_stack) > depth:
                qs.conclude_tracing()

    @custom_vmap
    def kernel(angles, state):
        return run_with_state(state, evaluate, angles, with_jacobian=with_jacobian)

    @kernel.def_vmap
    def batch_rule(axis_size, in_batched, angles, state):
        if in_batched != [True, False]:
            raise TracingModeError("Only the angles may be batched")
        if not axis_size:
            raise ValueError("Quantum batching requires a non-empty batch")
        value_shape = (axis_size, *readout_shape)
        # The loop always returns two arrays. For values alone, the Jacobian can be empty.
        jacobian_shape = (*value_shape, angles.shape[1] if with_jacobian else 0)

        def body(index, carry):
            values, jacobian = evaluate(angles[index], with_jacobian=with_jacobian)
            value_buffer, jacobian_buffer = carry
            return (
                dynamic_update_slice(value_buffer, values[None], (index, 0, 0)),
                dynamic_update_slice(jacobian_buffer, jacobian[None], (index, 0, 0, 0)),
            )

        # Frozen angles can leave a LinearizeTrace between batching and
        # the state. All quantum work belongs to the state's primal trace.
        with core.set_current_trace(state._trace):
            buffers = (
                jnp.zeros(value_shape, dtype=angles.dtype),
                jnp.zeros(jacobian_shape, dtype=angles.dtype),
            )
            batched_result = run_with_state(state, q_fori_loop, 0, axis_size, body, buffers)
        # Values and Jacobians have a batch axis. The loop passes one quantum state through all runs.
        return batched_result, ((True, True), False)

    result, qs.abs_qst = kernel(angles, state)
    return result


def _primal_batch_info(trace, state_trace, angles):
    """Return the vmap input and its batch axis when the Jasp state owns the computation."""
    if not isinstance(trace, BatchTrace):
        return None
    parent = trace.parent_trace
    primal, batch_axis = trace.to_batch_info(angles)
    if isinstance(parent, LinearizeTrace) and (not isinstance(primal, Tracer) or primal._trace is state_trace):
        # The angles are frozen under this reverse-mode trace.
        parent = parent.parent_trace
    if parent is state_trace:
        return primal, batch_axis
    return None


def _has_frozen_angles(trace, state_trace, angles):
    """Identify angles independent of the active reverse-mode input, possibly under vmap."""
    if not isinstance(trace, LinearizeTrace):
        return False
    parent = trace.parent_trace
    if parent is not state_trace and not (isinstance(parent, BatchTrace) and parent.parent_trace is state_trace):
        return False
    return not isinstance(angles, Tracer) or angles._trace is state_trace or angles._trace is parent


def differentiable_expectation(
    state_preparation: Callable,
    *,
    observable: Callable | tuple[Callable, ...] = Z,
    qubit_index: int | tuple[int, ...] = 0,
) -> Callable:
    r"""Create a Pauli readout with a first-order parameter-shift gradient for JAX.

    Parameters
    ----------
    state_preparation : callable
        Circuit that prepares a register from angles. The circuit and angles
        must meet the requirements of :func:`parameter_shift_jacobian`.
    observable : callable or tuple of callables, optional
        ``X``, ``Y`` or ``Z`` (default), or a non-empty tuple of these,
        known at tracing time.
    qubit_index : int or tuple of int, optional
        Qubit index or non-empty tuple of indices, known at tracing time. The default is 0.

    Returns
    -------
    callable
        Function from angles to a 2D array with the same dtype, one row per
        selected qubit and one column per observable. Inside ``jaspify`` or
        ``qjit``, it supports first-order reverse-mode differentiation and a
        single ``jax.vmap`` axis. A readout runs the circuit once. Computing
        its gradient adds two runs per angle.

    Examples
    --------
    >>> import jax
    >>> from qrisp.algorithms.qml.gradients import differentiable_expectation
    >>> from qrisp.algorithms.qml.templates import prepare_rotation
    >>> from qrisp.jasp import jaspify
    >>> readout = differentiable_expectation(prepare_rotation)
    >>> @jaspify
    ... def z_value_and_grad(angles):
    ...     return jax.value_and_grad(lambda a: readout(a)[0, 0])(angles)

    """
    if not callable(state_preparation):
        raise TypeError("state_preparation must be callable")

    readout_shape = (
        len(qubit_index) if isinstance(qubit_index, tuple) else 1,
        len(observable) if isinstance(observable, tuple) else 1,
    )

    def evaluate(angles, *, with_jacobian=False):
        qs = TracingQuantumSession.get_instance()
        if qs.abs_qst is None:
            raise TracingModeError("Use differentiable_expectation inside jaspify or qrisp.jasp.qjit")
        trace = core.trace_ctx.trace
        state_trace = qs.abs_qst._trace
        batch_info = _primal_batch_info(trace, state_trace, angles)
        if batch_info is not None:
            unbatched_angles, batch_axis = batch_info
            if batch_axis is None:
                # Every batch entry uses the same angles, so evaluate them once.
                with core.set_current_trace(state_trace):
                    return evaluate(unbatched_angles, with_jacobian=with_jacobian)
            return _batch_expectation(evaluate, angles, qs.abs_qst, readout_shape, with_jacobian=with_jacobian)
        if trace is not state_trace or (isinstance(angles, Tracer) and angles._trace is not state_trace):
            raise TracingModeError(
                "differentiable_expectation supports first-order reverse-mode AD and one vmap axis inside Jasp"
            )
        result = _parameter_shift_jasp(
            state_preparation,
            angles,
            observable,
            qubit_index,
            evaluation="value_and_jacobian" if with_jacobian else "value",
        )
        return jnp.asarray(result["readout"], dtype=angles.dtype), jnp.asarray(result["values"], dtype=angles.dtype)

    @custom_vjp
    def readout(angles):
        return evaluate(angles)[0]

    def forward(angles):
        # Save the Jacobian so the backward pass can reuse it without another circuit run.
        return evaluate(angles, with_jacobian=True)

    def backward(jacobian, cotangent):
        # Sum over qubits and observables to get one gradient per angle.
        return (jnp.tensordot(cotangent, jacobian, axes=((0, 1), (0, 1))),)

    readout.defvjp(forward, backward)

    def dispatch(angles):
        qs = TracingQuantumSession.get_instance()
        if qs.abs_qst is None:
            raise TracingModeError("Use differentiable_expectation inside jaspify or qrisp.jasp.qjit")
        trace = core.trace_ctx.trace
        state_trace = qs.abs_qst._trace
        if trace is state_trace or _primal_batch_info(trace, state_trace, angles) is not None:
            return evaluate(angles)[0]
        if _has_frozen_angles(trace, state_trace, angles):
            # These angles do not depend on the input being differentiated, so one readout is enough.
            with core.set_current_trace(trace.parent_trace):
                return evaluate(angles)[0]
        return readout(angles)

    return dispatch
