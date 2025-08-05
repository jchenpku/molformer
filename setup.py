# setup.py
from setuptools import setup, find_packages

setup(
    name="molformer",
    version="0.1.0",
    description="MolFormer: Large-Scale Chemical Language Representations",
    author="IBM Research",
    packages=find_packages(include=['training', 'finetune', 'notebooks', 'training.*', 'finetune.*', 'notebooks.*']),
    # The find_packages() function will automatically discover all your packages
    # (directories with an __init__.py file), so make sure they exist.
    install_requires=[
        # List exact versions from your environment setup
        "torch==1.7.1",
        "numpy==1.22.3",
        "pandas==1.2.4",
        "scikit-learn==0.24.2",
        "scipy==1.6.2",
        "rdkit-pypi==2022.03.2",
        "transformers==4.6.0",
        "pytorch-lightning==1.1.5",
        "pytorch-fast-transformers==0.4.0",
        "datasets==1.6.2",
        "pyarrow",
        "lmdb",
        "lancedb",
        "tqdm",
    ],
    python_requires='>=3.8',
)
