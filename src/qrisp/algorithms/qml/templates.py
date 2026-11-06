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

from qrisp import QuantumVariable, ry


def prepare_ry(angle: float) -> QuantumVariable:
    """Prepare a fresh single-qubit register with an RY rotation.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    QuantumVariable
        A new single-qubit register whose circuit prepares RY(angle)|0>.
        In eager mode, each call creates an independent quantum session.

    """
    qv = QuantumVariable(1)
    ry(angle, qv[0])
    return qv


def prepare_two_ry(angle_0: float, angle_1: float) -> QuantumVariable:
    """Prepare a single-qubit register with two consecutive RY rotations.

    Parameters
    ----------
    angle_0 : float
        Angle of the first rotation in radians.
    angle_1 : float
        Angle of the second rotation in radians.

    Returns
    -------
    QuantumVariable
        A new register whose circuit prepares RY(angle_1) RY(angle_0)|0>.
        In eager mode, each call creates an independent quantum session.

    """
    qv = QuantumVariable(1)
    ry(angle_0, qv[0])
    ry(angle_1, qv[0])
    return qv
