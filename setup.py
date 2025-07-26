#!/usr/bin/env python3
"""
Setup script for Price Predictor Project
========================================

Package configuration and installation script.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="price-predictor",
    version="1.0.0",
    author="Price Predictor Team",
    author_email="team@pricepredictor.com",
    description="A comprehensive stock price prediction system with API compliance",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/price-predictor",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/price-predictor/issues",
        "Documentation": "https://github.com/your-username/price-predictor/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
            "pre-commit>=2.0",
            "sphinx>=3.0",
            "jupyter>=1.0",
        ],
        "visualization": [
            "matplotlib>=3.0",
            "seaborn>=0.11",
            "plotly>=5.0",
            "dash>=2.0",
        ],
        "ml": [
            "tensorflow>=2.0",
            "torch>=1.0",
            "xgboost>=1.0",
            "lightgbm>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "price-predictor=scripts.data_pipeline:main",
            "train-model=scripts.train_model:main",
            "run-predictions=scripts.run_predictions:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.conf"],
    },
    zip_safe=False,
)
