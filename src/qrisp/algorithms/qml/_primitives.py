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

# Later gates use the state returned by the check, so they run after it.
assert_condition_p = QuantumPrimitive("assert_condition")


@assert_condition_p.def_abstract_eval
def assert_condition_abstract_eval(condition, state, *, message):
    """Check the abstract signature of a runtime assertion."""
    if not isinstance(condition, ShapedArray) or condition.shape != () or condition.dtype != bool:
        raise TypeError("A runtime assertion requires a scalar boolean condition")
    if not isinstance(state, AbstractQuantumState) or not isinstance(message, str):
        raise TypeError("A runtime assertion requires a quantum state and a string message")
    return AbstractQuantumState()


@assert_condition_p.def_impl
def assert_condition_implementation(condition, state, *, message):
    """Raise on a failed condition while preserving the quantum state."""
    if not condition:
        raise ValueError(message)
    return state


expectation_z_p = QuantumPrimitive("expectation_z")
expectation_z_p.multiple_results = True


@expectation_z_p.def_abstract_eval
def expectation_z_abstract_eval(qubits, _qc, *, qubit_indices):
    """Return the shape of the local Z expectations and the quantum state."""
    if not isinstance(qubits, AbstractQubitArray):
        raise TypeError("Analytic Z readout requires a qubit array")
    if not qubit_indices or any(type(index) is not int or index < 0 for index in qubit_indices):
        raise ValueError("Analytic Z readout requires non-negative integer indices")
    return ShapedArray((len(qubit_indices),), dtypes.canonicalize_dtype(np.float64)), AbstractQuantumState()


@expectation_z_p.def_impl
def expectation_z_implementation(qubits, qc, *, qubit_indices):
    """Evaluate local Z expectations without projecting the simulated state."""
    if isinstance(qc, QuantumCircuit):
        raise NotImplementedError("Analytic Z readout cannot be extracted as a QuantumCircuit")
    if qc.simulator != "qrisp":
        raise NotImplementedError("Analytic Z readout requires the Qrisp statevector simulator")
    if any(not 0 <= index < len(qubits) for index in qubit_indices):
        raise ValueError("qubit_index must satisfy 0 <= index < qv.size")

    qc.apply_buffer()
    values = []
    for index in qubit_indices:
        physical_index = qc.qubit_to_index_dict[qubits[index]]
        outcomes, probabilities = qc.quantum_state.multi_measure([physical_index])
        values.append(np.sum((1 - 2 * outcomes) * probabilities, dtype=np.float64))
    return np.asarray(values, dtype=dtypes.canonicalize_dtype(np.float64)), qc
