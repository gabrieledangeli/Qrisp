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

"""Parameter-shift derivatives for small QML circuits."""

from math import pi

from qrisp.algorithms.qml.execution import expectation_z
from qrisp.algorithms.qml.templates import prepare_ry, prepare_two_ry


def ry_z_derivative(angle: float) -> float:
    """Differentiate the Z expectation of RY(angle)|0> with respect to angle.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    float
        The derivative, equal to -sin(angle) up to simulator precision.

    Notes
    -----
    For this single-rotation circuit, the parameter-shift rule gives
    (F(angle + pi/2) - F(angle - pi/2)) / 2, where F is the Z expectation.
    The two evaluations use independently prepared registers.

    """
    shift = pi / 2
    value_plus = expectation_z(prepare_ry(angle + shift))
    value_minus = expectation_z(prepare_ry(angle - shift))
    return (value_plus - value_minus) / 2


def two_ry_z_derivatives(angle_0: float, angle_1: float) -> tuple[float, float]:
    """Differentiate the Z expectation with respect to each of two RY angles.

    Parameters
    ----------
    angle_0 : float
        Angle of the first rotation in radians.
    angle_1 : float
        Angle of the second rotation in radians.

    Returns
    -------
    tuple of float
        Partial derivatives with respect to angle_0 and angle_1, in that order.
        Both equal -sin(angle_0 + angle_1) up to simulator precision.

    Notes
    -----
    Each parameter-shift evaluation changes only one rotation angle and
    creates a fresh register. If both angles equal a shared weight theta,
    the derivative with respect to theta is the sum of the two partials.

    """
    shift = pi / 2

    value_0_plus = expectation_z(prepare_two_ry(angle_0 + shift, angle_1))
    value_0_minus = expectation_z(prepare_two_ry(angle_0 - shift, angle_1))
    derivative_0 = (value_0_plus - value_0_minus) / 2

    value_1_plus = expectation_z(prepare_two_ry(angle_0, angle_1 + shift))
    value_1_minus = expectation_z(prepare_two_ry(angle_0, angle_1 - shift))
    derivative_1 = (value_1_plus - value_1_minus) / 2

    return derivative_0, derivative_1
