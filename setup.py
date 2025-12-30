"""
Setup script for coulomb_mc package.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="coulomb_mc",
    version="0.1.0",
    description="3D Radiation Monte Carlo with ML Integration",
    author="William Comaskey",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "numba>=0.58",
        "h5py>=3.8",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=12.0"],
        "ml": ["torch>=2.0", "tensorboard>=2.12"],
        "dev": ["pytest>=7.3", "black>=23.0", "mypy>=1.3", "ipython>=8.12"],
        "all": ["cupy-cuda11x>=12.0", "torch>=2.0", "pytest>=7.3"],
    },
)
