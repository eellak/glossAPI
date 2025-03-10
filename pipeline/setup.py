from setuptools import setup, find_packages
import os
import glob

# Find all model files
model_files = glob.glob('src/glossapi/models/*.joblib')

# Get long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="glossapi",
    version="0.0.3.5.2",
    author="GlossAPI Team",
    author_email="foivos@example.com",
    description="A library for processing academic texts in Greek and other languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eellak/glossAPI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "glossapi": ["models/*.joblib"],
    },
    data_files=[
        ('models', model_files),
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "dask",
        "pyarrow",
    ],
    include_package_data=True,
)
