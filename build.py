import glob
import hashlib
import json
import os
import platform
import shutil

import numpy as np
from setuptools import Extension

# Cache file to store information about the last build
CACHE_FILE = ".build_cache.json"


def get_file_hash(filename):
    """Calculate the hash of a file to detect changes"""
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash


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
        ),
        # Clustering modules
        Extension(
            "pypamm.clustering.cluster_utils",
            ["src/pypamm/clustering/cluster_utils.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        # Lib modules
        Extension(
            "pypamm.lib._opx",
            ["src/pypamm/lib/_opx.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pypamm.lib.distance",
            ["src/pypamm/lib/distance.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        # Density modules
        Extension(
            "pypamm.density.kde",
            ["src/pypamm/density/kde.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]

    print(f"Defined {len(extensions)} extension modules")
    return extensions


def should_rebuild(extensions):
    """Check if any source files have changed since the last build"""
    if not os.path.exists(CACHE_FILE):
        print("No build cache found, will compile all extensions")
        return True

    try:
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Invalid or missing cache file, will compile all extensions")
        return True

    # Check if any source files have changed
    for ext in extensions:
        for source in ext.sources:
            current_hash = get_file_hash(source)
            if source not in cache or cache[source] != current_hash:
                print(f"Source file {source} has changed, will recompile")
                return True

    # Check if any .pxd files have changed
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".pxd"):
                filepath = os.path.join(root, file)
                current_hash = get_file_hash(filepath)
                if filepath not in cache or cache[filepath] != current_hash:
                    print(f"Header file {filepath} has changed, will recompile")
                    return True

    print("No source files have changed, skipping compilation")
    return False


def update_cache(extensions):
    """Update the cache with the current file hashes"""
    cache = {}

    # Add all source files to the cache
    for ext in extensions:
        for source in ext.sources:
            cache[source] = get_file_hash(source)

    # Add all .pxd files to the cache
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".pxd"):
                filepath = os.path.join(root, file)
                cache[filepath] = get_file_hash(filepath)

    # Write the cache to disk
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def build(setup_kwargs):
    """Function called by Poetry during the build process"""
    print("Starting Poetry build process...")

    # Import Cython here to ensure it's installed
    from Cython.Build import cythonize

    # Get the extension modules
    extensions = create_extensions()

    # Always rebuild when called by Poetry
    # This is safer and avoids potential issues with the build process
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                extensions,
                compiler_directives={
                    "language_level": 3,
                    "boundscheck": False,
                    "wraparound": False,
                },
            ),
            "include_dirs": [np.get_include()],
        }
    )

    print("Added extension modules to setup_kwargs")
    return setup_kwargs


# If run as a script, build the extensions directly
if __name__ == "__main__":
    print("Running build.py as a script")

    # Clean up any previous build artifacts
    if os.path.exists("build"):
        print("Cleaning up previous build directory")
        shutil.rmtree("build")

    # Ensure the output directories exist
    os.makedirs("pypamm", exist_ok=True)

    # Create directories for all modules
    for ext in create_extensions():
        module_parts = ext.name.split(".")
        if len(module_parts) > 2:  # For submodules like pypamm.clustering.xxx
            submodule = module_parts[-2]
            os.makedirs(f"pypamm/{submodule}", exist_ok=True)

    # Import Cython here to ensure it's installed
    from distutils.core import setup

    from Cython.Build import cythonize

    # Get the extension modules
    extensions = create_extensions()

    # Check if we need to rebuild
    rebuild = should_rebuild(extensions)

    if rebuild:
        # Build the extensions
        ext_modules = cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
            },
        )

        # Build the extensions in place
        setup(
            name="pypamm",
            ext_modules=ext_modules,
            include_dirs=[np.get_include()],
            script_args=["build_ext", "--inplace"],
        )

        # Copy the built extensions to the src directory
        for ext in extensions:
            module_parts = ext.name.split(".")
            if len(module_parts) > 2:  # For submodules like pypamm.clustering.xxx
                module_name = module_parts[-1]
                submodule = module_parts[-2]
                source_path = f"pypamm/{submodule}/{module_name}.*.so"
                target_dir = f"src/pypamm/{submodule}"
            else:  # For direct modules like pypamm.xxx
                module_name = module_parts[-1]
                source_path = f"pypamm/{module_name}.*.so"
                target_dir = "src/pypamm"

            # Find the built extension file
            built_files = glob.glob(source_path)
            if built_files:
                source_file = built_files[0]
                target_file = os.path.join(target_dir, os.path.basename(source_file))
                print(f"Copying {source_file} to {target_file}")
                shutil.copy(source_file, target_file)

        # Update the cache after successful compilation
        update_cache(extensions)
    else:
        print("Skipping compilation as no source files have changed")

    print("Build complete")
