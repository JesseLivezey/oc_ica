"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='oc_ica',
    description='OC_ICA models and analysis.',
    long_description=long_description,
    author='Jesse Livezey and Alex Bujan',
    author_email='jesse.livezey@berkeley.edu',
    install_requires = [
      'h5py',
      'numpy',
      'scipy',
      'theano']
    )
