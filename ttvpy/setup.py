#import os, sys

#from distutils.core import setup, Extension
#from distutils import sysconfig
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


class custom_build_ext(build_ext):
    def build_extensions(self):
        # Override the compiler executables. Importantly, this
        # removes the "default" compiler flags that would
        # otherwise get passed on to to the compiler, i.e.,
        # distutils.sysconfig.get_var("CFLAGS").
        self.compiler.set_executable("compiler_so", "g++")
        self.compiler.set_executable("compiler_cxx", "g++")
        self.compiler.set_executable("linker_so", "g++")
        build_ext.build_extensions(self)

#OpenMP
#compile_args = ['-std=c++17','-O3','-fopenmp','-fPIC','-DUSE_OPENBLAS']
#include_dirs = ['../include','../../include','pybind11/include','/usr/include/x86_64-linux-gnu/openblas64-pthread']
#libraries=['openblas','m']
#extra_link_args=['-fopenmp']

#IntelMKL
compile_args = ['-std=c++17','-O3','-fopenmp','-fPIC','-DUSE_MKLBLAS', '-DMKL_ILP64', '-m64']
include_dirs = ['../include','../../include','pybind11/include','/usr/include/mkl']
libraries=['m','dl','iomp5']
extra_link_args=['-Wl,--start-group','/usr/lib/x86_64-linux-gnu/libmkl_intel_ilp64.a','/usr/lib/x86_64-linux-gnu/libmkl_intel_thread.a','/usr/lib/x86_64-linux-gnu/libmkl_core.a','-Wl,--end-group','-fopenmp']


name='ttvpy'
sources=['src/wrapped_ttv.cpp']


ext_modules = [
  Pybind11Extension(
    name,
    sources,
    include_dirs=include_dirs,
    libraries=libraries,
    extra_link_args=extra_link_args,
    language='c++',
    extra_compile_args=compile_args,
    ),
]

setup(
    name='ttvpy',
    version='0.0.2',
    author='Cem Bassoy',
    author_email='cem.bassoy@gmail.com',
    description='Python module for fast tensor-times-vector multiplication',
    ext_modules=ext_modules,
    python_requires=">=3.8",
    build_cmd = {"build_ext": custom_build_ext},
    zip_safe=False,
    install_requires=['numpy','pybind11'], 
)
