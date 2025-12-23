from setuptools import setup, Extension
import pybind11
import platform
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def find_arrow_paths():
    conda_prefix = os.environ.get('CONDA_PREFIX')

    if conda_prefix:
        arrow_include = os.path.join(conda_prefix, 'include')
        arrow_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(arrow_include) and os.path.exists(arrow_lib):
            return arrow_include, arrow_lib

    try:
        arrow_include = subprocess.check_output(
            ['pkg-config', '--variable=includedir', 'arrow'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        arrow_lib = subprocess.check_output(
            ['pkg-config', '--variable=libdir', 'arrow'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if arrow_include and arrow_lib:
            return arrow_include, arrow_lib
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    system = platform.system()
    if system == 'Darwin':
        machine = platform.machine()
        search_paths = ['/opt/homebrew', '/usr/local'] if machine == 'arm64' else ['/usr/local', '/opt/homebrew']
    elif system == 'Linux':
        search_paths = ['/usr', '/usr/local']
    elif system == 'Windows':
        search_paths = ['C:\\arrow', os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Arrow')]
    else:
        search_paths = ['/usr/local']

    for prefix in search_paths:
        arrow_include = os.path.join(prefix, 'include')
        arrow_lib = os.path.join(prefix, 'lib')
        if os.path.exists(os.path.join(arrow_include, 'arrow')) and os.path.exists(arrow_lib):
            return arrow_include, arrow_lib

    logger.error("Apache Arrow library not found. Install with: conda install -c conda-forge arrow-cpp pyarrow")
    sys.exit(1)

arrow_include, arrow_lib = find_arrow_paths()

common_compile_args = ['-std=c++17', '-O3']
extra_link_args = []

if platform.system() == 'Darwin':
    common_compile_args.append('-mmacosx-version-min=10.15')
    extra_link_args = [f'-Wl,-rpath,{arrow_lib}']
elif platform.system() == 'Linux':
    extra_link_args = [f'-Wl,-rpath,{arrow_lib}']
elif platform.system() == 'Windows':
    common_compile_args = ['/std:c++17', '/O2', '/EHsc']

csv_module = Extension(
    'fasttfidf_csv',
    sources=['fasttfidf_csv.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=common_compile_args,
)

parquet_module = Extension(
    'fasttfidf_parquet',
    sources=['fasttfidf_parquet.cpp'],
    include_dirs=[
        pybind11.get_include(),
        arrow_include,
    ],
    library_dirs=[arrow_lib],
    libraries=['arrow', 'parquet'],
    language='c++',
    extra_compile_args=common_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='fasttfidf',
    version='1.0.1',
    author='Jaskaran Singh Puri',
    description='High-performance TF-IDF with SIMD optimization',
    ext_modules=[csv_module, parquet_module],
    install_requires=[
        'pybind11>=2.12.0',
        'numpy>=1.19.0',
        'scipy>=1.5.0'
    ],
    python_requires='>=3.9',
    zip_safe=False,
)