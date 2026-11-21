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

"""Small state-preparation circuits for quantum machine learning."""

from collections.abc import Callable

from qrisp import QuantumVariable, rx, ry, rz
from qrisp.algorithms.qml._primitives import assert_condition_p
from qrisp.jasp import TracingModeError, TracingQuantumSession, check_for_tracing_mode
from qrisp.typing import FloatLike, NDArrayLike


def prepare_rotation(
    angle: FloatLike | list[FloatLike] | tuple[FloatLike, ...] | NDArrayLike,
    *,
    rotation: Callable = ry,
    qv: QuantumVariable | None = None,
) -> QuantumVariable:
    r"""Apply one rotation per qubit in a Jasp program.

    Parameters
    ----------
    angle : scalar, list, tuple or array
        Angle in radians for one qubit, or a non-empty 1D sequence with one
        angle per qubit. The number of angles must be known at tracing time.
    rotation : callable, optional
        Choose ``rx``, ``ry`` (default) or ``rz`` at tracing time.
    qv : QuantumVariable, optional
        Live register in the active trace, with one qubit per angle. Its size
        check does not support inversion or quantum control.

    Returns
    -------
    QuantumVariable
        The rotated register. If ``qv`` is omitted, the function allocates a new register.

    Examples
    --------
    >>> from qrisp import measure, rx
    >>> from qrisp.algorithms.qml.templates import prepare_rotation
    >>> from qrisp.jasp import jaspify
    >>> @jaspify
    ... def circuit(theta, phi):
    ...     qv = prepare_rotation(theta)
    ...     prepare_rotation(phi, rotation=rx, qv=qv)
    ...     return measure(qv[0])

    """
    # Jasp needs an active session to record the gates.
    qs = TracingQuantumSession.get_instance()
    if qs.abs_qst is None or not check_for_tracing_mode():
        raise TracingModeError("Use prepare_rotation inside jaspify or qrisp.jasp.qjit")
    
    # Check that the quantum state belongs to the current JAX trace.
    qs._check_in_scope()

    if not any(rotation is gate for gate in (rx, ry, rz)):
        raise TypeError("rotation must be one of the Qrisp functions rx, ry or rz")

    # Put even a single angle in a list so we can loop over it.
    angles = _normalize_angles(angle)

    if qv is None:
        # Start with one qubit in |0> for each angle.
        qv = QuantumVariable(len(angles))
    else:
        # To reuse a register, check that it belongs to this trace and has one qubit per angle.
        _check_rotation_register(qv, qs, len(angles))

    # The first angle rotates qubit 0, the second rotates qubit 1, and so on.
    for index, value in enumerate(angles):
        rotation(value, qv[index])
    return qv


def _normalize_angles(angle):
    """Collect one scalar per qubit, preserving traced angle values."""
    shape_error = "angle must be a scalar or a non-empty one-dimensional sequence"

    # A single angle goes through the same loop as a sequence of angles.
    is_vector = isinstance(angle, NDArrayLike) and angle.ndim == 1
    if is_vector or isinstance(angle, (list, tuple)):
        items = angle
    else:
        items = (angle,)

    angles = []
    for value in items:
        if isinstance(value, NDArrayLike) and value.ndim == 0:
            # Read the scalar in a way JAX can trace.
            value = value[()]

        # Each qubit gets one angle, so reject nested inputs like [[a, b]].
        is_nested_array = isinstance(value, NDArrayLike) and value.ndim != 0
        if isinstance(value, (list, tuple)) or is_nested_array:
            raise ValueError(shape_error)
        angles.append(value)

    # A rotation needs at least one angle.
    if not angles:
        raise ValueError(shape_error)
    return angles


def _check_rotation_register(qv: QuantumVariable, qs: TracingQuantumSession, num_angles: int) -> None:
    """Check that the register can be reused for one rotation per qubit."""
    if not isinstance(qv, QuantumVariable):
        raise TypeError("qv must be a QuantumVariable or None")

    # Nested traces share a session, so check both the session and the trace.
    belongs_to_active_trace = qv.qs is qs and qv.reg.tracer._trace is qs.abs_qst._trace
    if not belongs_to_active_trace:
        raise TracingModeError("qv must belong to the active Jasp trace")

    # Use identity here because "=="" compares the quantum values.
    if any(deleted_qv is qv for deleted_qv in qs.deleted_qv_list):
        raise ValueError("Cannot apply rotations to a deleted QuantumVariable")

    # The size may only be known at runtime. Record the check before the rotations.
    qs.abs_qst = assert_condition_p.bind(
        qv.size == num_angles,
        qs.abs_qst,
        message="The number of angles must match qv.size",
    )
