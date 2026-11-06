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

"""Single-example squared error and its prediction derivative."""


def squared_error(prediction: float, target: float) -> float:
    """Compute the squared error for a single scalar prediction.

    Parameters
    ----------
    prediction : float
        Predicted value.
    target : float
        Desired value.

    Returns
    -------
    float
        (prediction - target)**2, without averaging over a batch.

    """
    return (prediction - target) ** 2


def squared_error_derivative(prediction: float, target: float) -> float:
    """Differentiate the squared error with respect to the prediction.

    Parameters
    ----------
    prediction : float
        Predicted value.
    target : float
        Desired value, held constant when differentiating.

    Returns
    -------
    float
        2 * (prediction - target). This is the derivative with respect to
        prediction; a weight derivative also requires the prediction's
        derivative with respect to that weight.

    """
    return 2 * (prediction - target)
