"""Python wrapper for the cython implementation of Madgwick filter.

Sources:
    - https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    - Madgwick et al. (https://doi.org/10.1109/ICORR.2011.5975346)
    - https://ahrs.readthedocs.io for the wrapper
    - G. Dugué's Rcpp wrapper for Madgwick algorithm.

Author: Romain Fayat, June 2020
"""
import numpy as np
from ahrs.filters import Madgwick as Madgwick_ahrs
from ahrs.common.orientation import acc2q, am2q
from madgwick.madgwick_cython import CprodQv, madgwickIMU, madgwickAHRS


class Madgwick(Madgwick_ahrs):
    """Wrapper for Cython implementation of the Madgwick filter.

    The API and documentation are directly taken from the ahrs module.
    """

    __doc__ += "\n" + Madgwick_ahrs.__doc__

    def _compute_all(self):
        "Compute all quaternions from the input data."
        self._compute_q0()  # initial quaternion
        self._input_sanity_check()  # sanity check on the inputs
        # IMU pipeline if no mag data was provided
        if self.mag is None:
            return madgwickIMU(self.acc, self.gyr, self.q0,
                               self.gain, self.frequency)[:-1]
        # MARG pipeline if mag data was provided
        else:
            return madgwickAHRS(self.acc, self.gyr, self.mag, self.q0,
                                self.gain, self.frequency)[:-1]

    def _compute_q0(self):
        "Compute the initial quaternion for the iterative algorithm."
        if self.q0 is not None:
            self.q0 /= np.linalg.norm(self.q0)
        elif self.mag is None:
            self.q0 = acc2q(self.acc[0])
        else:
            self.q0 = am2q(self.acc[0], self.mag[0])

    def _input_sanity_check(self):
        "Sanity check on the input data (matching shapes, no missing values)."
        # Make sure the shapes of the input data are consistent
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag is not None and self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        # Check for nan values in the input data (not supported at the moment)
        if np.any(np.isnan(self.acc)):
            raise ValueError("nan accelerometer values are not supported yet")
        if np.any(np.isnan(self.gyr)):
            raise ValueError("nan gyroscope values are not supported yet")
        if self.mag is not None and np.any(np.isnan(self.mag)):
            raise ValueError("nan gyroscope values are not supported yet")

    def gravity_estimate(self):
        "Estimate the coordinates of gravity in the sensor reference frame."
        if not hasattr(self, "Q"):
            raise ValueError(
                "The object was not instantiated with at least accelerometer and gyroscope data."  # noqa E501
            )
        return CprodQv(self.Q, np.array([0, 0, 1], dtype=float))
