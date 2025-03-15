import os
import sys
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

def build(setup_kwargs):
    # Define the extension modules
    extensions = [
        Extension(
            "pypamm.grid_selection",
            ["src/pypamm/grid_selection.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
        ),
        Extension(
            "pypamm.neighbor_graph",
            ["src/pypamm/neighbor_graph.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
        )
    ]
    
    # Add the extension modules to the setup_kwargs
    setup_kwargs.update({
        "ext_modules": cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
            }
        ),
        "include_dirs": [np.get_include()]
    })
    
    return setup_kwargs 