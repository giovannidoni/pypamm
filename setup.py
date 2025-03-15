# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['pypamm']

package_data = \
{'': ['*']}

install_requires = \
['cython>=3.0.0,<4.0.0', 'numpy>=2.0.0,<3.0.0']

setup_kwargs = {
    'name': 'pypamm',
    'version': '0.1.0',
    'description': 'Probabilistic Analysis of Molecular Motifs',
    'long_description': '',
    'author': 'Your Name',
    'author_email': 'your.email@example.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.12,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
