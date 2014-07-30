

from numpy.distutils.core import setup, Extension


import os


blas_build = 'internal'  # Options: intel, internal

source = ['dgbmv.pyf']

# Set up the libraries for building against BLAS. Either use an internal copy
# of dgbmv.f, or try and build against MKL.
if blas_build == 'intel':
    blas_lib = ['mkl_rt', 'iomp5', 'pthread', 'm']  # MKL libraries we need to link against
    blas_libdir = [os.environ['MKLROOT']+'/lib/intel64' if 'MKLROOT' in os.environ else '']
elif blas_build == 'internal':
    source += ['dgbmv.f']
    blas_lib = []
    blas_libdir = []

use_omp = True
omp_args = ['-fopenmp'] if use_omp else []

dgbmv_ext = Extension('dgbmv', source,
                      library_dirs=blas_libdir, libraries=blas_lib,
                      extra_compile_args=omp_args,
                      extra_link_args=omp_args)


setup(
    name='pfb-inverse',
    ext_modules=[dgbmv_ext]
)
