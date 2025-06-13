from setuptools import setup, Extension
import pybind11

eigen_include = "/home/u80856195/git/eigen"

ext_modules = [
    Extension(
        "pseudo_inverse",
        sources=["pseudo_inverse.cpp", "pseudo_inverse_bindings.cpp"],
        include_dirs=[pybind11.get_include(), eigen_include],
        language="c++",
    ),
]

setup(
    name="pseudo_inverse",
    version="0.1",
    ext_modules=ext_modules,
)