import os
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

def build(setup_kwargs):
    print("Starting build process...")
    # Define the extension modules
    extensions = [
        Extension(
            "pypamm.distance_metrics",
            ["src/pypamm/distance_metrics.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
        ),
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
    
    print(f"Defined {len(extensions)} extension modules")
    
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
    
    print("Added extension modules to setup_kwargs")
    
    return setup_kwargs

# If run as a script, build the extensions directly
if __name__ == "__main__":
    print("Running build.py as a script")
    setup_kwargs = {}
    build(setup_kwargs)
    
    # We don't need to build in place when running as a script
    # The Poetry build process will handle the installation
    print("Build complete") 