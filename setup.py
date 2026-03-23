from setuptools import setup, Extension
from pathlib import Path
import pybind11

eigen_include = Path(__file__).resolve().parent / "SBFVAR" / "third_party" / "eigen"

ext_modules = [
    Extension(
        "SBFVAR.cholcov.cholcov_module",
        sources=[
            "SBFVAR/cholcov/cholcov.cpp",
            "SBFVAR/cholcov/cholcov_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "SBFVAR.inverse.matrix_inversion",
        sources=["SBFVAR/inverse/matrix_inversion.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "SBFVAR.pseudo_inverse.pseudo_inverse",
        sources=[
            "SBFVAR/pseudo_inverse/pseudo_inverse.cpp",
            "SBFVAR/pseudo_inverse/pseudo_inverse_bindings.cpp",
        ],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
    Extension(
        "SBFVAR.solve.solve",
        sources=["SBFVAR/solve/solve.cpp"],
        include_dirs=[pybind11.get_include(), str(eigen_include)],
        language="c++",
    ),
]

setup(ext_modules=ext_modules)
