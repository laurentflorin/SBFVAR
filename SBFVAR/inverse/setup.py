from setuptools import setup, Extension
import pybind11

eigen_include = "/home/u80856195/git/eigen"

ext_modules = [
    Extension(
        "matrix_inversion",
        sources=["matrix_inversion.cpp"],
        include_dirs=[pybind11.get_include(), eigen_include],
        language="c++",
    ),
]

setup(
    name="matrix_inversion",
    version="0.1",
    ext_modules=ext_modules,
)