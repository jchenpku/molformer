# /workspace/molformer/setup.py
from setuptools import setup, find_packages

setup(
    name="molformer",
    version="0.1.0",
    description="MolFormer: Large-Scale Chemical Language Representations",
    author="IBM Research",
    # find_packages() will discover all directories with an __init__.py file
    packages=find_packages(),
    #
    # By leaving out 'install_requires', we tell pip:
    # "Just make this code importable, and assume all dependencies
    # have already been installed by the user (e.g., via Docker)."
    #
    python_requires='>=3.8',
)
