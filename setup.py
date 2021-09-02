"Handle the processing of cython code and the installation of madgwick."
try:
    import numpy as np
    from Cython.Build import cythonize
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "Cython and numpy are required for performing the installation.\n"
        "Please run:\n"
        "    `$ pip install numpy Cython`\n"
        "or\n"
        "    `pip install -r requirements.txt`\n"
        "before running the installation pipeline."
    )
import os
from distutils.core import setup


def read(fname):
    "Read a file in the current directory."
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="madgwick",
    author="Romain Fayat",
    version="0.1",
    author_email="r.fayat@gmail.com",
    description="Cython implementation of the Madgwick filter.",
    ext_modules=cythonize("madgwick/madgwick_cython.pyx"),
    include_dirs=[np.get_include()],
    install_requires=["numpy", "Cython", "ahrs"],
    packages=["madgwick"],
    long_description=read('README.md')
)
