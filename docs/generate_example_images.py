#!/usr/bin/env python
"""
Generate example output images for the PyPAMM documentation.

This script runs all the example scripts and copies the generated images to the docs/images directory.
"""

import os
import subprocess
import shutil
from pathlib import Path

# Create the docs/images directory if it doesn't exist
images_dir = Path("docs/images")
images_dir.mkdir(exist_ok=True)

# List of example scripts to run
example_scripts = [
    "grid_selection_example.py",
    "neighbor_graph_example.py",
    "quick_shift_example.py",
    "mst_example.py",
    "pipeline_example.py"
]

# Run each example script and copy the generated images
for script in example_scripts:
    print(f"Running {script}...")
    
    # Run the script using Poetry
    result = subprocess.run(
        ["poetry", "run", "python", f"examples/{script}"],
        capture_output=True,
        text=True
    )
    
    # Print the output
    print(result.stdout)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error running {script}:")
        print(result.stderr)
        continue
    
    # Copy the generated images to the docs/images directory
    for image_file in Path(".").glob("*.png"):
        # Create a new filename based on the script name
        new_name = f"{script.replace('.py', '')}_{image_file.name}"
        dest_path = images_dir / new_name
        
        # Copy the image
        shutil.copy(image_file, dest_path)
        print(f"Copied {image_file} to {dest_path}")
        
        # Remove the original image
        image_file.unlink()
        print(f"Removed {image_file}")

print("Done generating example images!") 