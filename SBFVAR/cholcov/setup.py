from setuptools import setup, Extension
import pybind11

eigen_include = "/home/u80856195/git/eigen"

ext_modules = [
    Extension(
        "cholcov",
        sources=["cholcov.cpp", "cholcov_bindings.cpp"],
        include_dirs=[pybind11.get_include(), eigen_include],
        language="c++",
    ),
]

setup(
    name="cholcov",
    version="0.1",
    ext_modules=ext_modules,
)