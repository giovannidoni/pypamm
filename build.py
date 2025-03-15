import os
import sys
import platform
from setuptools import Extension
import numpy as np
import shutil
import glob

def create_extensions():
    """Create extension modules with platform-specific settings"""
    print("Creating extension modules...")
    
    # Define platform-specific compiler flags
    extra_compile_args = ["-O3", "-ffast-math", "-Wno-unreachable-code"]
    
    # Add platform-specific flags
    if platform.system() == "Linux":
        # Disable vectorization on Linux to avoid undefined symbol errors
        extra_compile_args.extend(["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION", "-fno-tree-vectorize"])
    
    # Define the extension modules
    extensions = [
        Extension(
            "pypamm.distance_metrics",
            ["src/pypamm/distance_metrics.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pypamm.grid_selection",
            ["src/pypamm/grid_selection.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pypamm.neighbor_graph",
            ["src/pypamm/neighbor_graph.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pypamm.quick_shift",
            ["src/pypamm/quick_shift.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pypamm.mst",
            ["src/pypamm/mst.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ]
    
    print(f"Defined {len(extensions)} extension modules")
    return extensions

def build(setup_kwargs):
    """Function called by Poetry during the build process"""
    print("Starting Poetry build process...")
    
    # Import Cython here to ensure it's installed
    from Cython.Build import cythonize
    
    # Get the extension modules
    extensions = create_extensions()
    
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
    
    # Ensure the output directory exists
    os.makedirs("pypamm", exist_ok=True)
    
    # Import Cython here to ensure it's installed
    from Cython.Build import cythonize
    from distutils.core import setup
    
    # Get the extension modules
    extensions = create_extensions()
    
    # Build the extensions
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        }
    )
    
    # Build the extensions in place
    setup(
        name="pypamm",
        ext_modules=ext_modules,
        include_dirs=[np.get_include()],
        script_args=["build_ext", "--inplace"]
    )
    
    # Copy the built extensions to the src directory
    for ext in extensions:
        module_name = ext.name.split(".")[-1]
        source_path = f"pypamm/{module_name}.*.so"
        
        # Find the built extension file
        built_files = glob.glob(source_path)
        if built_files:
            target_dir = "src/pypamm"
            source_file = built_files[0]
            target_file = os.path.join(target_dir, os.path.basename(source_file))
            print(f"Copying {source_file} to {target_file}")
            shutil.copy(source_file, target_file)
    
    print("Build complete") 