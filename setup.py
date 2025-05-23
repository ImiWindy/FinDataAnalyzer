"""Setup script for FinDataAnalyzer package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="findataanalyzer",
    version="0.1.0",
    author="FinDataAnalyzer Team",
    author_email="info@findataanalyzer.com",
    description="A financial data analysis package with prediction capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/findataanalyzer/findataanalyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "dash>=2.0.0",
        "plotly>=5.3.0",
        "pyyaml>=6.0",
        "requests>=2.26.0",
        "sqlalchemy>=1.4.0",
        "python-multipart>=0.0.5",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
            "pre-commit>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "findataanalyzer=findataanalyzer.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 