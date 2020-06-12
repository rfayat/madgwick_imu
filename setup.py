import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
setup(name="madgwick",
      ext_modules=cythonize("madgwick/madgwick_cython.pyx"),
      include_dirs=[np.get_include()])
