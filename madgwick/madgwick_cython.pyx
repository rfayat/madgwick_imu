cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Cprodqv(np.ndarray[DTYPE_t, ndim=1] q,
             np.ndarray[DTYPE_t, ndim=1] v):
   "Rotate a vector v (ndarray, (3,)) by a quaternion q (ndarray, (4,))"
   cdef double r1,r2,r3 # elements of the rotated vector
   cdef double q0,q1,q2,q3,q0q0,q0q1,q0q2,q0q3,q1q1,q1q2,q1q3,q2q2,q2q3,q3q3

   # grab the quaternion's value
   q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

   # Auxiliary variables to avoid repeated arithmetic
   q0q0, q0q1, q0q2, q0q3 = 2*q0*q0, 2*q0*q1, 2*q0*q2, 2*q0*q3
   q1q1, q1q2, q1q3       =          2*q1*q1, 2*q1*q2, 2*q1*q3
   q2q2, q2q3             =                   2*q2*q2, 2*q2*q3
   q3q3                   =                            2*q3*q3

   # rotate the vector
   r1 = (q0q0-1+q1q1) * v[0] +  (q1q2+q0q3)  * v[1] +  (q1q3-q0q2) * v[2]
   r2 =  (q1q2-q0q3)  * v[0] + (q0q0-1+q2q2) * v[1] +  (q2q3+q0q1) * v[2]
   r3 =  (q1q3+q0q2)  * v[0] +  (q2q3-q0q1)  * v[1] + (q0q0-1+q3q3)* v[2]

   return(r1, r2, r3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Cprodqq(np.ndarray[DTYPE_t, ndim=1] q,
             np.ndarray[DTYPE_t, ndim=1] r):
    "Quaternion product q * r where q and r are two np arrays with 4 elements."
    cdef double s0, s1, s2, s3 # elements of the output quaternion
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]

    s0 = q0*r0 - q1*r1 - q2*r2 - q3*r3
    s1 = q0*r1 + q1*r0 + q2*r3 - q3*r2
    s2 = q0*r2 - q1*r3 + q2*r0 + q3*r1
    s3 = q0*r3 + q1*r2 - q2*r1 + q3*r0

    return(s0, s1, s2, s3)

@cython.boundscheck(False)
@cython.wraparound(False)
def CprodQv(np.ndarray[DTYPE_t, ndim=2] Q,
            np.ndarray[DTYPE_t, ndim=1] v):
  """
  Rotate a 3D vector by an array of quaternions.

  INPUT
  *v, ndarray (3,): vector to rotate
  *Q, ndarray (nQuaternion, 4): quaternions to use to rotate v.

  RETURN
  *rotatedVectors (nQuaternion, 3): rotated vectors
  """
  cdef int nQuaternion = Q.shape[0] # total number of quaternions used
  cdef np.ndarray[DTYPE_t, ndim=2] rotatedVectors = np.zeros((nQuaternion, 3), dtype=np.float) # array of rotated vectors
  cdef int quaternionIndex

  for quaternionIndex in range(nQuaternion):
    quaternion = Q[quaternionIndex]
    rotatedVectors[quaternionIndex] = Cprodqv(quaternion, v)

  return(rotatedVectors)

# -------------------------------------------------------
#                       Madgwick IMU
# -------------------------------------------------------
cdef gradientMadgwickIMU(double q0, double q1, double q2, double q3,
                         double ax, double ay, double az):
  """
  Compute the gradient decent algorithm corrective step for Madgwick's IMU algorithm.
  """
  cdef double s0, s1, s2, s3
  cdef double _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3
  # Auxiliary variables to avoid repeated arithmetic
  _2q0 = 2.0 * q0
  _2q1 = 2.0 * q1
  _2q2 = 2.0 * q2
  _2q3 = 2.0 * q3
  _4q0 = 4.0 * q0
  _4q1 = 4.0 * q1
  _4q2 = 4.0 * q2
  _8q1 = 8.0 * q1
  _8q2 = 8.0 * q2
  q0q0 = q0 * q0
  q1q1 = q1 * q1
  q2q2 = q2 * q2
  q3q3 = q3 * q3

  # Gradient decent algorithm corrective step
  s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
  s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
  s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
  s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
  recipNorm = 1.0/sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)  # normalise step magnitude
  s0 *= recipNorm
  s1 *= recipNorm
  s2 *= recipNorm
  s3 *= recipNorm

  return(s0, s1, s2, s3)

# @cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef madgwickIMUStep(np.ndarray[DTYPE_t, ndim=1] accStep,
                    np.ndarray[DTYPE_t, ndim=1] gyrStep,
                    np.ndarray[DTYPE_t, ndim=1] quatStep,
                    double beta,
                    double sampleFreq):
  "Step for the IMU algorithm (could be used for real-time computation)."
  cdef double gx, gy, gz, ax, ay, az, q0, q1, q2, q3
  cdef double qDot1, qDot2, qDot3, qDot4
  cdef double recipNorm
  cdef double s0, s1, s2, s3

  # Grab acc, gyr and quat values
  gx, gy, gz = gyrStep[0], gyrStep[1], gyrStep[2]
  ax, ay, az = accStep[0], accStep[1], accStep[2]
  q0, q1, q2, q3 = quatStep[0], quatStep[1], quatStep[2], quatStep[3]

  # Rate of change of quaternion from gyroscope
  qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
  qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
  qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
  qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

  if ax!=0 or ay!=0 or az!=0:
    # Normalise accelerometer measurement
    recipNorm = 1.0/sqrt(ax * ax + ay * ay + az * az)
    ax *= recipNorm
    ay *= recipNorm
    az *= recipNorm

    # Gradient descent algorithm corrective step
    s0, s1, s2, s3 = gradientMadgwickIMU(q0, q1, q2, q3, ax, ay, az)

    # Apply feedback step
    qDot1 -= beta * s0
    qDot2 -= beta * s1
    qDot3 -= beta * s2
    qDot4 -= beta * s3

  # Integrate rate of change of quaternion
  q0 += qDot1 * (1.0 / sampleFreq)
  q1 += qDot2 * (1.0 / sampleFreq)
  q2 += qDot3 * (1.0 / sampleFreq)
  q3 += qDot4 * (1.0 / sampleFreq)

  # Normalise quaternion
  recipNorm = 1.0/sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
  quatStep[0] = q0 * recipNorm
  quatStep[1] = q1 * recipNorm
  quatStep[2] = q2 * recipNorm
  quatStep[3] = q3 * recipNorm

  return(quatStep)

@cython.boundscheck(False)
@cython.wraparound(False)
def madgwickIMU(np.ndarray[DTYPE_t, ndim=2] acc,
                np.ndarray[DTYPE_t, ndim=2] gyr,
                np.ndarray[DTYPE_t, ndim=1] quat0,
                double beta,
                double sampleFreq):
  cdef int n_row  = acc.shape[0]
  cdef int row = 0
  cdef np.ndarray[DTYPE_t, ndim=2] quat = np.zeros((n_row+1, 4), dtype=np.float)
  quat[0,:] = quat0

  for row in range(n_row):
    quatStep = quat[row]
    quat[row+1] = madgwickIMUStep(acc[row], gyr[row], quatStep, beta, sampleFreq)

  return(quat)

# -------------------------------------------------------
#                       Madgwick AHRS
# -------------------------------------------------------
cdef gradientMadgwickAHRS(double q0, double q1, double q2, double q3,
                          double ax, double ay, double az,
                          double mx, double my, double mz):
  """
  Compute the gradient decent algorithm corrective step for Madgwick's AHRS algorithm.
  """
  cdef double s0, s1, s2, s3
  cdef double _2q0mx, _2q0my, _2q0mz, _2q1mx, _2q0, _2q1, _2q2, _2q3, _2q0q2, _2q2q3
  cdef double q0q0, q0q1, q0q2, q0q3, q1q1, q1q2, q1q3, q2q2, q2q3, q3q3
  cdef double hx, hy, _2bx, _2bz, _4bx,  _4bz

  # Auxiliary variables to avoid repeated arithmetic
  _2q0mx = 2.0 * q0 * mx
  _2q0my = 2.0 * q0 * my
  _2q0mz = 2.0 * q0 * mz
  _2q1mx = 2.0 * q1 * mx
  _2q0 = 2.0 * q0
  _2q1 = 2.0 * q1
  _2q2 = 2.0 * q2
  _2q3 = 2.0 * q3
  _2q0q2 = 2.0 * q0 * q2
  _2q2q3 = 2.0 * q2 * q3
  q0q0 = q0 * q0
  q0q1 = q0 * q1
  q0q2 = q0 * q2
  q0q3 = q0 * q3
  q1q1 = q1 * q1
  q1q2 = q1 * q2
  q1q3 = q1 * q3
  q2q2 = q2 * q2
  q2q3 = q2 * q3
  q3q3 = q3 * q3

  # Reference direction of Earth's magnetic field
  hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3
  hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3
  _2bx = sqrt(hx * hx + hy * hy)
  _2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3
  _4bx = 2.0 * _2bx
  _4bz = 2.0 * _2bz

  # Gradient decent algorithm corrective step
  s0 = -_2q2 * (2.0 * q1q3 - _2q0q2 - ax) + _2q1 * (2.0 * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
  s1 = _2q3 * (2.0 * q1q3 - _2q0q2 - ax) + _2q0 * (2.0 * q0q1 + _2q2q3 - ay) - 4.0 * q1 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az) + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
  s2 = -_2q0 * (2.0 * q1q3 - _2q0q2 - ax) + _2q3 * (2.0 * q0q1 + _2q2q3 - ay) - 4.0 * q2 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
  s3 = _2q1 * (2.0 * q1q3 - _2q0q2 - ax) + _2q2 * (2.0 * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)

  # normalise step magnitude and return step
  recipNorm = 1.0/sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
  s0 *= recipNorm
  s1 *= recipNorm
  s2 *= recipNorm
  s3 *= recipNorm
  return(s0, s1, s2, s3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef madgwickAHRSStep(np.ndarray[DTYPE_t, ndim=1] accStep,
                      np.ndarray[DTYPE_t, ndim=1] gyrStep,
                      np.ndarray[DTYPE_t, ndim=1] magStep,
                      np.ndarray[DTYPE_t, ndim=1] quatStep,
                      double beta, double sampleFreq):
  # Preallocation
  cdef double gx, gy, gz, ax, ay, az, mx, my, mz, q0, q1, q2, q3
  cdef double qDot1, qDot2, qDot3, qDot4
  cdef double recipNorm
  cdef double s0, s1, s2, s3

  # Grab acc, gyr, mag and quat values
  gx, gy, gz = gyrStep[0], gyrStep[1], gyrStep[2]
  ax, ay, az = accStep[0], accStep[1], accStep[2]
  mx, my, mz = magStep[0], magStep[1], magStep[2]
  q0, q1, q2, q3 = quatStep[0], quatStep[1], quatStep[2], quatStep[3]

  # Rate of change of quaternion from gyroscope
  qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
  qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
  qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
  qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

  if ax!=0 or ay!=0 or az!=0:
    # Normalise accelerometer measurement
    recipNorm = 1.0/sqrt(ax * ax + ay * ay + az * az)
    ax *= recipNorm
    ay *= recipNorm
    az *= recipNorm

    # Normalise magnetometer measurement
    recipNorm = 1.0/sqrt(mx * mx + my * my + mz * mz)
    mx *= recipNorm
    my *= recipNorm
    mz *= recipNorm

    # Gradient descent algorithm corrective step
    s0, s1, s2, s3 = gradientMadgwickAHRS(q0, q1, q2, q3, ax, ay, az, mx, my, mz)

    # Apply feedback step
    qDot1 -= beta * s0
    qDot2 -= beta * s1
    qDot3 -= beta * s2
    qDot4 -= beta * s3

  # Integrate rate of change of quaternion
  q0 += qDot1 * (1.0 / sampleFreq)
  q1 += qDot2 * (1.0 / sampleFreq)
  q2 += qDot3 * (1.0 / sampleFreq)
  q3 += qDot4 * (1.0 / sampleFreq)

  # Normalise quaternion
  recipNorm = 1.0/sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
  quatStep[0] = q0 * recipNorm
  quatStep[1] = q1 * recipNorm
  quatStep[2] = q2 * recipNorm
  quatStep[3] = q3 * recipNorm

  return(quatStep)

@cython.boundscheck(False)
@cython.wraparound(False)
def madgwickAHRS(np.ndarray[DTYPE_t, ndim=2] acc,
                np.ndarray[DTYPE_t, ndim=2] gyr,
                np.ndarray[DTYPE_t, ndim=2] mag,
                np.ndarray[DTYPE_t, ndim=1] quat0,
                double beta, double sampleFreq):
  cdef int n_row  = acc.shape[0]
  cdef int row = 0
  cdef np.ndarray[DTYPE_t, ndim=2] quat = np.zeros((n_row+1, 4), dtype=np.float)
  quat[0,:] = quat0

  # Iteratively update the value of the quaternion and store it in quat
  for row in range(n_row):
    quatStep = quat[row]
    quat[row+1] = madgwickAHRSStep(acc[row], gyr[row], mag[row], quatStep, beta, sampleFreq)

  return(quat)
