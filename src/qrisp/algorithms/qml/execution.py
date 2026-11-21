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

"""Jasp evaluation of local Pauli expectation values."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from qrisp import QuantumVariable, h, s, s_dg
from qrisp.algorithms.qml._primitives import assert_condition_p, expectation_z_p
from qrisp.jasp import TracingModeError, TracingQuantumSession
from qrisp.operators.qubit import X, Y, Z


def expectation_pauli(
    qv: QuantumVariable,
    *,
    observable: Callable | tuple[Callable, ...] = Z,
    qubit_index: int | tuple[int, ...] = 0,
) -> dict:
    r"""Compute analytic local Pauli expectations while preserving the register state.

    Parameters
    ----------
    qv : QuantumVariable
        Live register in the active Jasp program.
    observable : callable or tuple of callables, optional
        ``X``, ``Y`` or ``Z`` from ``qrisp.operators.qubit``, or a non-empty tuple
        of these. The selection must be known at tracing time. The default is ``Z``.
    qubit_index : int or tuple of int, optional
        Qubit index or non-empty tuple of indices, known at tracing time. The default is 0.

    Returns
    -------
    dict
        ``values`` is a 2D JAX array with one row per qubit and one column per
        observable. The tuples ``qubit_indices`` and ``observables`` label the axes
        in the requested order, including repeats. Return the numeric values
        from functions passed to ``jaspify`` or ``qjit``.

    Examples
    --------
    >>> from qrisp.algorithms.qml.execution import expectation_pauli
    >>> from qrisp.algorithms.qml.templates import prepare_rotation
    >>> from qrisp.jasp import jaspify
    >>> @jaspify
    ... def readout(theta):
    ...     qv = prepare_rotation(theta)
    ...     return expectation_pauli(qv)["values"][0, 0]

    """
    if not isinstance(qv, QuantumVariable):
        raise TypeError("qv must be a QuantumVariable")

    # Jasp needs an active session to record the readout.
    qs = TracingQuantumSession.get_instance()
    if qs.abs_qst is None or qv.qs is not qs:
        raise TracingModeError("Use expectation_pauli inside jaspify or qrisp.jasp.qjit with a tracing register")

    # Put single observables and indices in tuples so we can loop over them.
    observables = _normalize_observables(observable)
    indices = _normalize_qubit_indices(qubit_index)

    # Check that the quantum state belongs to the current JAX trace.
    qs._check_in_scope()

    if any(deleted_qv is qv for deleted_qv in qs.deleted_qv_list):
        raise ValueError("Cannot evaluate a deleted QuantumVariable")

    if X in observables or Y in observables:
        # X and Y need gates before the readout. Check the indices before those gates run.
        qs.abs_qst = assert_condition_p.bind(
            qv.size > max(indices),
            qs.abs_qst,
            message="qubit_index must satisfy 0 <= index < qv.size",
        )

    # Read each distinct observable once, even if the caller asks for it twice.
    columns_by_pauli = {}
    for pauli in dict.fromkeys(observables):
        columns_by_pauli[pauli] = _expectation_column(qv, pauli, indices)

    # Stack the columns in the requested order. Each row belongs to one requested qubit.
    ordered_columns = [columns_by_pauli[pauli] for pauli in observables]
    return {
        "values": jnp.stack(ordered_columns, axis=1),
        "qubit_indices": indices,
        "observables": tuple(pauli.__name__ for pauli in observables),
    }


def _normalize_observables(observable: Callable | tuple[Callable, ...]) -> tuple[Callable, ...]:
    # A single Z becomes (Z,). A tuple like (Y, X, Y) keeps its order and repeats.
    if isinstance(observable, tuple):
        observables = observable
    else:
        observables = (observable,)

    if not observables:
        raise ValueError("observable must contain at least one Pauli constructor")

    # The caller passes the Qrisp functions themselves, such as X.
    for pauli in observables:
        is_supported = pauli is X or pauli is Y or pauli is Z
        if not is_supported:
            raise TypeError("observable must be X, Y, Z or a tuple of these Qrisp constructors")

    return observables


def _normalize_qubit_indices(qubit_index: int | tuple[int, ...]) -> tuple[int, ...]:
    # Jasp needs to know which qubits to read when it traces the circuit.
    if qubit_index is None:
        raise TracingModeError("Specify explicit qubit indices for expectation_pauli in Jasp")

    if isinstance(qubit_index, tuple):
        indices = qubit_index
    else:
        indices = (qubit_index,)

    # Check the types before calling int(), which would also accept floats and strings.
    for index in indices:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("qubit_index must contain Python or NumPy integer scalars")

    # Indices start at zero. The upper bound is checked at runtime using the register size.
    if not indices or any(index < 0 for index in indices):
        raise ValueError("qubit_index must select at least one index with 0 <= index < qv.size")

    # Convert NumPy integers to Python integers and keep the requested order.
    return tuple(int(index) for index in indices)


def _expectation_column(qv: QuantumVariable, pauli: Callable, indices: tuple[int, ...]):
    """Build one column of expectation values for the selected qubits."""
    if pauli is Z:
        # The primitive can read Z on all requested qubits in one call.
        return _expectation_z(qv, indices)

    values_by_index = {}
    for index in dict.fromkeys(indices):
        qubit = qv[index]

        # For X, apply H before reading Z. For Y, apply S-dagger first, then H.
        if pauli is Y:
            s_dg(qubit)
        h(qubit)
        values_by_index[index] = _expectation_z(qv, (index,))[0]

        # Undo the gates before the next read. Two indices can refer to the same qubit.
        h(qubit)
        if pauli is Y:
            s(qubit)

    return jnp.stack([values_by_index[index] for index in indices])


def _expectation_z(qv: QuantumVariable, indices: tuple[int, ...]):
    """Return one Z expectation per qubit index, in the requested order."""
    session = qv.qs

    # Record the readout in Jasp. Each value is P(0) - P(1) for one qubit.
    z_values, next_state = expectation_z_p.bind(
        qv.reg.tracer,
        session.abs_qst,
        qubit_indices=indices,
    )

    # Pass the returned state to later gates so they run after the readout.
    session.abs_qst = next_state
    return z_values
