from setuptools import setup, find_packages

setup(
    name="merlion-benchmarking",
    version="0.1.0",
    description="A flexible benchmarking framework for Merlion time series models.",
    author="Christoph Eder",
    author_email="christoph.eder@ntnu.no",
    python_requires=">=3.10",
    packages=find_packages(where="src"),  # <-- find all packages under src/
    package_dir={"": "src"},
    install_requires=[
        "salesforce-merlion>=2.0.4",
        "scikit-learn>=1.4",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0"
    ],
)
