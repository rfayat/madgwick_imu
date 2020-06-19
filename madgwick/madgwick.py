"""
Python wrapper for the cython implementation of Madgwick's algorithm.

TODO:
    - Clean tests
    - Benchmark versus numba JIT compilation (would make the install more
      straightforward)

Author: Romain Fayat

Sources:
    - G. Dugué's R wrapper for Madgwick algorithm.
    - https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from madgwick.madgwick_cython import CprodQv, madgwickIMU, madgwickAHRS

quat0 = np.array([1, 0, 0, 0], dtype=np.float)
initialGravity = np.array([0, 0, 1], dtype=np.float)


def computeGravity(acc, gyr, mag=None, sampleFreq=300.,
                   beta=.1, quat0=quat0, initialGravity=initialGravity):
    """
    Inputs
    ------
    acc, array (n_timestamps, 3)
        Accelerometer data (x, y, z) in G.

    gyr, array (n_timestamps, 3)
        Gyroscope data (x, y, z) in RADIANS.

    mag, array (n_timestamps, 3)
        Magnetometer data (x, y, z) in Tesla.

    sampleFreq, float (default: 300.)
        Sampling frequency.

    beta, float (default: .1)
        Beta parameter for the adaptive filter

    quat0, array (4) (default: array([1., 0., 0., 0.]))
        Initial quaternion for the iterative rotation estimate.

    initialGravityn array (3) (default: array([0., 0., 1.])
        Initial gravity value. The default value correspond to a vertical
        vector.

    Returns
    -------
    gravityEstimate, array (n_timestamps, 3)
        Estimate of the coordinates of the gravity orientation (in G).

    """
    if mag is None:
        allRotations = madgwickIMU(acc, gyr, quat0, beta, sampleFreq)
    else:
        allRotations = madgwickAHRS(acc, gyr, mag, quat0, beta, sampleFreq)

    # Gravity estimate by applying the rotations to a vertical vector
    gravityEstimate = CprodQv(allRotations, initialGravity)

    # Remove the last point to obtain a length constistent with the input
    gravityEstimate = gravityEstimate[:-1, :]
    return gravityEstimate
