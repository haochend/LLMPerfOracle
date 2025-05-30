"""Setup script for LLMPerfOracle."""

from setuptools import find_packages, setup

setup(
    name="llmperforacle",
    version="0.1.0",
    description="A virtualized environment for comparative performance analysis of LLM serving frameworks",
    author="LLMPerfOracle Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "simpy>=4.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scipy>=1.10",
        "click>=8.1",
    ],
    entry_points={
        "console_scripts": [
            "llmperforacle=llmperforacle.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)