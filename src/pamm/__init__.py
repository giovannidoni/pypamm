"""
PAMM (Probabilistic Analysis of Molecular Motifs) package.
"""

try:
    from .grid_selection import select_grid_points
    __all__ = ['select_grid_points']
except ImportError:
    import warnings
    warnings.warn("Could not import grid_selection module. Make sure Cython extensions are compiled.")
    __all__ = []
