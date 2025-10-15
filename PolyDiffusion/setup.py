#!/usr/bin/env python
"""Setup script for PolyDiffusion package."""

from pathlib import Path
from setuptools import find_packages, setup

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="PolyDiffusion",
    version="0.1.0",
    description="Discrete diffusion transformer for guided homopolymer generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Polygenesis Team",
    author_email="",
    url="https://github.com/your-org/PolyDiffusion",
    packages=find_packages(where=".", include=["PolyDiffusion*"]),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "chemistry": [
            "rdkit>=2022.09.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "polydiff-vocab=PolyDiffusion.scripts.build_vocab:main",
            "polydiff-sample=PolyDiffusion.scripts.sample_cli:main",
            "polydiff-eval=PolyDiffusion.scripts.evaluate_stage:main",
            "polydiff-train-a=PolyDiffusion.src.train.train_stage_a:main",
            "polydiff-train-b=PolyDiffusion.src.train.train_stage_b:main",
            "polydiff-train-c=PolyDiffusion.src.train.train_stage_c:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="diffusion, transformers, polymers, chemistry, generative-models",
    project_urls={
        "Bug Reports": "https://github.com/your-org/PolyDiffusion/issues",
        "Source": "https://github.com/your-org/PolyDiffusion",
    },
)
