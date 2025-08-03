from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# python setup.py build_ext --inplace

ext_modules = [
    Pybind11Extension(
        "retrograde_cpp",
        ["boilerplate.cpp", "calc.cpp"],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="retrograde_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)