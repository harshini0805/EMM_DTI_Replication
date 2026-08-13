"""
Setup configuration for EMM-DTI package.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="emm-dti",
    version="0.1.0",
    author="EMM-DTI Development Team",
    description="Enhanced Mamba-Based Model for Drug-Target Interaction Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emm-dti",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "rdkit>=2023.09.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipython>=8.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emm-dti-train=emm_dti.train:main",
            "emm-dti-eval=emm_dti.evaluate:main",
        ],
    },
)
