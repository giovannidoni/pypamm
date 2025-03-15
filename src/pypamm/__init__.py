"""
PyPAMM (Python Probabilistic Analysis of Molecular Motifs) package.
"""

try:
    from .grid_selection import select_grid_points
    from .neighbor_graph import build_neighbor_graph
    __all__ = ['select_grid_points', 'build_neighbor_graph']
except ImportError:
    import warnings
    warnings.warn("Could not import modules. Make sure Cython extensions are compiled.")
    __all__ = []
