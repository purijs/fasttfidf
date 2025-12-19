from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess
import platform

__version__ = '1.0.0'

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def check_avx2():
    """Check if CPU supports AVX2"""
    if platform.machine() in ['x86_64', 'AMD64']:
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            return 'avx2' in result.stdout.lower()
        except:
            return False
    return False

def check_neon():
    """Check if CPU supports NEON"""
    if platform.machine() in ['aarch64', 'arm64']:
        return True
    return False

extra_compile_args = ['-std=c++17', '-O3']
extra_link_args = []

if check_avx2():
    print("AVX2 support detected - enabling SIMD optimizations")
    extra_compile_args.append('-mavx2')
elif check_neon():
    print("ARM NEON support detected - enabling SIMD optimizations")
else:
    print("No SIMD support detected - using scalar implementation")

ext_modules = [
    Extension(
        'fasttfidf',
        ['fasttfidf.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='fasttfidf',
    version=__version__,
    author='Jaskaran S. Puri',
    author_email='jaskaranpuri@gmail.com',
    url='https://github.com/purijs/fasttfidf',
    description='Lightning-fast TF-IDF for datasets that don\'t fit in RAM',
    long_description='High-performance TF-IDF vectorizer with SIMD optimization, zero-copy streaming, and automatic memory management for massive datasets.',
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.12.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0'
    ],
    setup_requires=['pybind11>=2.12.0'],  # ← Also update here
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.6',
)