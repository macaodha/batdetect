from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from sys import platform

extra_compile_args = []
extra_link_args = []

try:
    from Cython.Distutils.build_ext import build_ext
except ImportError:
    print('Error: Cython not installed. please install by running "conda install cython". exiting')
    exit()

if platform == "linux" or platform == "linux2":
    # linux
    extra_compile_args.append('-fopenmp')
    extra_compile_args.append('-ffast-math')
    extra_compile_args.append('-msse')
    extra_compile_args.append('-msse2')
    extra_compile_args.append('-msse3')
    extra_compile_args.append('-msse4')
    extra_compile_args.append('-s')
    extra_compile_args.append('-std=c99')
    extra_link_args.append('-fopenmp')

elif platform == "darwin":
    # OS X
    extra_compile_args.append('-fopenmp')
    extra_compile_args.append('-ffast-math')
    extra_compile_args.append('-msse')
    extra_compile_args.append('-msse2')
    extra_compile_args.append('-msse3')
    extra_compile_args.append('-msse4')
    extra_compile_args.append('-s')
    extra_compile_args.append('-std=c99')
    extra_link_args.append('-fopenmp')

    import os
    os.environ["CC"] = "gcc-6"
    os.environ["CXX"] = "gcc-6"
elif platform == "win32":
    # Windows
    pass

extensions = [
    Extension("nms", ["nms.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)
    ]

setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
    )
