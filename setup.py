

from numpy.distutils.core import setup, Extension


import os


blas_build = 'internal'

source = ['dgbmv.pyf']

if blas_build == 'intel':
    # Set library includes (taking into account which MPI library we are using)."
    blas_lib = ['mkl_rt', 'iomp5', 'pthread', 'm']
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
