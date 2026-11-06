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

"""Statevector-based evaluation of single-qubit Z expectations."""

import numpy as np

from qrisp import QuantumVariable
from qrisp.jasp import TracingModeError, TracingQuantumSession


def expectation_z(qv: QuantumVariable) -> float:
    """Compute the Pauli Z expectation of a prepared single-qubit register.

    Parameters
    ----------
    qv : QuantumVariable
        A live register containing the only qubit in its session. State
        preparation must use unitary operations and have no open environments.

    Returns
    -------
    float
        The expectation P(0) - P(1), computed without shot sampling and subject
        to the numerical precision of Qrisp's statevector simulator.

    Raises
    ------
    TracingModeError
        If the register belongs to a Jasp tracing session.
    ValueError
        If the register or its circuit does not satisfy the restrictions above.

    Examples
    --------
    Prepare RY(theta)|0> and compare its Z expectation with cos(theta).

    >>> import numpy as np
    >>> from qrisp import QuantumVariable, ry
    >>> from qrisp.algorithms.qml.execution import expectation_z
    >>> theta = 0.37
    >>> qv = QuantumVariable(1)
    >>> ry(theta, qv[0])
    >>> result = expectation_z(qv)
    >>> bool(np.isclose(result, np.cos(theta), atol=1e-6, rtol=0))
    True

    """
    if isinstance(qv.qs, TracingQuantumSession):
        raise TracingModeError("expectation_z only supports eager execution")
    if qv.size != 1:
        raise ValueError("expectation_z requires a single-qubit register")
    if qv.is_deleted():
        raise ValueError("Cannot evaluate a deleted QuantumVariable")
    if qv.qs.env_stack:
        raise ValueError("Close all quantum environments before evaluating expectation_z")
    if list(qv.reg) != list(qv.qs.qubits):
        raise ValueError("The register must contain the only qubit in its session")

    # Expand composite gates so that non-unitary operations are also checked.
    qc = qv.qs.transpile()
    for instruction in qc.data:
        if instruction.op.name in ("measure", "reset", "qb_dealloc") or instruction.clbits:
            raise ValueError("expectation_z does not support measurements, resets or qubit reuse")

    statevector = qc.statevector_array()
    probabilities = np.abs(statevector) ** 2
    return float(probabilities[0] - probabilities[1])
