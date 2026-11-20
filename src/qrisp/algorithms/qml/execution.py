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

    qs = TracingQuantumSession.get_instance()
    if qs.abs_qst is None or qv.qs is not qs:
        raise TracingModeError("Use expectation_pauli inside jaspify or qrisp.jasp.qjit with a tracing register")

    observables = observable if isinstance(observable, tuple) else (observable,)
    if not observables:
        raise ValueError("observable must contain at least one Pauli constructor")

    for pauli in observables:
        if not any(pauli is constructor for constructor in (X, Y, Z)):
            raise TypeError("observable must be X, Y, Z or a tuple of these Qrisp constructors")

    indices = _normalize_qubit_indices(qubit_index)

    qs._check_in_scope()

    if any(deleted_qv is qv for deleted_qv in qs.deleted_qv_list):
        raise ValueError("Cannot evaluate a deleted QuantumVariable")

    if X in observables or Y in observables:
        # Check bounds before applying the first basis-change gate.
        qs.abs_qst = assert_condition_p.bind(
            qv.size > max(indices),
            qs.abs_qst,
            message="qubit_index must satisfy 0 <= index < qv.size",
        )

    # Read each observable once, then reuse its column if it appears again.
    columns = {}
    for pauli in dict.fromkeys(observables):
        columns[pauli] = _expectation_column(qv, pauli, indices)
    return {
        "values": jnp.stack([columns[pauli] for pauli in observables], axis=1),
        "qubit_indices": indices,
        "observables": tuple(pauli.__name__ for pauli in observables),
    }


def _normalize_qubit_indices(qubit_index: int | tuple[int, ...]) -> tuple[int, ...]:
    """Return static indices as Python integers for the Jasp primitive."""
    if qubit_index is None:
        raise TracingModeError("Specify explicit qubit indices for expectation_pauli in Jasp")

    indices = qubit_index if isinstance(qubit_index, tuple) else (qubit_index,)
    for index in indices:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("qubit_index must contain Python or NumPy integer scalars")

    if not indices or any(index < 0 for index in indices):
        raise ValueError("qubit_index must select at least one index with 0 <= index < qv.size")

    return tuple(int(index) for index in indices)


def _expectation_column(qv, pauli, indices):
    """Evaluate one local Pauli observable on the requested qubits."""
    # The primitive reads Z directly. For X, apply H first. For Y, apply S-dagger then H.
    if pauli is Z:
        return _expectation_z(qv, indices)

    values = {}
    for index in dict.fromkeys(indices):
        # Restore the basis after each readout because distinct indices can alias the same qubit.
        qubit = qv[index]
        if pauli is Y:
            s_dg(qubit)
        h(qubit)
        values[index] = _expectation_z(qv, (index,))[0]
        h(qubit)
        if pauli is Y:
            s(qubit)
    return jnp.stack([values[index] for index in indices])


def _expectation_z(qv, indices):
    """Record an analytic Z readout and update the traced quantum state."""
    values, qv.qs.abs_qst = expectation_z_p.bind(qv.reg.tracer, qv.qs.abs_qst, qubit_indices=indices)
    return values
