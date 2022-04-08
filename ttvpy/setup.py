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

#g++ -Wall -shared -std=c++17 src/wrapped_ttv.cpp -o ttvpy.so $(python3 -m pybind11 --includes) -I../include -fPIC -fopenmp -DUSE_OPENBLAS -lm -lopenblas
# python3 setup.py clean --all && rm -rf __pycache__ ttvpy.cpython-38-x86_64-linux-gnu.so build/
# python3 setup.py build_ext -i
# sudo pip install -e .
# python3 -m unittest discover -v

compile_args = ['-std=c++17','-O3','-fopenmp','-fPIC','-DUSE_OPENBLAS'] # '-Wall', 
include_dirs = ['../include','../../include','pybind11/include'] #  '../../../include', '../../include', 
name='ttvpy'
sources=['src/wrapped_ttv.cpp']
libraries=['openblas','m'] # '-lm', ,'-lopenblas'
extra_link_args=['-fopenmp']

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
