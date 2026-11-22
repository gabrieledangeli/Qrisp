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

"""Jasp primitives used by the QML algorithms."""

import numpy as np
from jax import dtypes
from jax.core import ShapedArray

from qrisp.circuit import QuantumCircuit
from qrisp.jasp import AbstractQuantumState, AbstractQubitArray, QuantumPrimitive

# The next gate gets the quantum state only if the check passes.
assert_condition_p = QuantumPrimitive("assert_condition")


@assert_condition_p.def_abstract_eval
def assert_condition_abstract_eval(condition, state, *, message):
    """Check the types Jasp sees while tracing the assertion."""
    # Shape () means one value. The check needs a single True or False.
    is_scalar_bool = isinstance(condition, ShapedArray) and condition.shape == () and condition.dtype == bool
    if not is_scalar_bool:
        raise TypeError("A runtime assertion requires a scalar boolean condition")

    if not isinstance(state, AbstractQuantumState) or not isinstance(message, str):
        raise TypeError("A runtime assertion requires a quantum state and a string message")
    return AbstractQuantumState()


@assert_condition_p.def_impl
def assert_condition_implementation(condition, state, *, message):
    """Pass the state to the next operation if the runtime check succeeds."""
    if not condition:
        raise ValueError(message)
    return state


expectation_z_p = QuantumPrimitive("expectation_z")
# The caller receives one array of Z values and the state for the next operation.
expectation_z_p.multiple_results = True


@expectation_z_p.def_abstract_eval
def expectation_z_abstract_eval(qubits, _qc, *, qubit_indices):
    """Describe the two outputs before the circuit runs."""
    if not isinstance(qubits, AbstractQubitArray):
        raise TypeError("Analytic Z readout requires a qubit array")

    # The indices are known while tracing. The register size can depend on a runtime input.
    if not qubit_indices or any(type(index) is not int or index < 0 for index in qubit_indices):
        raise ValueError("Analytic Z readout requires non-negative integer indices")

    # Each requested index gets one entry, even if the same index appears twice.
    values_shape = (len(qubit_indices),)
    values_dtype = dtypes.canonicalize_dtype(np.float64)
    return ShapedArray(values_shape, values_dtype), AbstractQuantumState()


@expectation_z_p.def_impl
def expectation_z_implementation(qubits, qc, *, qubit_indices):
    """Read local Z expectations from the probabilities in the Qrisp simulator."""
    if isinstance(qc, QuantumCircuit):
        raise NotImplementedError("Analytic Z readout cannot be extracted as a QuantumCircuit")
    if qc.simulator != "qrisp":
        raise NotImplementedError("Analytic Z readout requires the Qrisp statevector simulator")

    if any(not 0 <= index < len(qubits) for index in qubit_indices):
        raise ValueError("qubit_index must satisfy 0 <= index < qv.size")

    # Apply pending gates so the probabilities include everything before the readout.
    qc.apply_buffer()
    values = []
    for index in qubit_indices:
        # Find this register qubit in the simulator's full state.
        qubit = qubits[index]
        simulator_index = qc.qubit_to_index_dict[qubit]
        outcomes, probabilities = qc.quantum_state.multi_measure([simulator_index])

        # Z gives +1 for outcome 0 and -1 for outcome 1. Weight each by its probability.
        z_eigenvalues = 1 - 2 * outcomes
        expectation = np.sum(z_eigenvalues * probabilities, dtype=np.float64)
        values.append(expectation)

    values_dtype = dtypes.canonicalize_dtype(np.float64)
    return np.asarray(values, dtype=values_dtype), qc


def _assert_condition_catalyst(condition, state, *, message):
    """Record the same runtime check in the Catalyst program."""
    from catalyst.jax_primitives import assert_p

    assert_p.bind(condition, error=message)
    return state


def _expectation_z_catalyst(qubits, state, *, qubit_indices):
    """Read the local Z expectations with Catalyst's PauliZ observable."""
    import jax.numpy as jnp
    from catalyst.jax_primitives import assert_p, expval_p, namedobs_p, qextract_p, qinsert_p

    register, free_positions = state
    # The counter gives the size of the local qubit array.
    indices_in_range = qubits.counter > max(qubit_indices)
    assert_p.bind(indices_in_range, error="qubit_index must satisfy 0 <= index < qv.size")

    values = []
    for index in qubit_indices:
        # Map the local index to a position in the full Catalyst register.
        position = qubits[index]
        qubit = qextract_p.bind(register, position)
        pauli_z = namedobs_p.bind(qubit, kind="PauliZ")
        values.append(expval_p.bind(pauli_z))

        # Put the qubit back in the same position for later operations.
        register = qinsert_p.bind(register, position, qubit)

    values_dtype = dtypes.canonicalize_dtype(np.float64)
    return jnp.asarray(values, dtype=values_dtype), (register, free_positions)


# The Catalyst interpreter uses these functions to translate the two Jasp primitives.
assert_condition_p.catalyst_lowering = _assert_condition_catalyst
expectation_z_p.catalyst_lowering = _expectation_z_catalyst
