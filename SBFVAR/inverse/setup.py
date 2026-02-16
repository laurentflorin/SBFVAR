from pathlib import Path
from setuptools import setup, Extension
import pybind11

# Use vendored Eigen headers to avoid system-level dependency.
eigen_include = Path(__file__).resolve().parent.parent / "third_party" / "eigen"

ext_modules = [
    Extension(
        "matrix_inversion",
        sources=["matrix_inversion.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
]

setup(
    name="matrix_inversion",
    version="0.1",
    ext_modules=ext_modules,
)