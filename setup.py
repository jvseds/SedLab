from setuptools import setup, find_packages

setup(
    name="grainsize",
    version="0.1.0",
    description="Sedimentological data‐analysis tools (grain‐size, XRF, forams, bryozoans)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jarden Aaltonen",
    url="https://github.com/jvseds/SedLab",
    license="MIT",
    include_package_data=True,

    packages=find_packages(
        exclude=[
            ".venv",
            "data-files",
            "notebooks",
            "tests",
            "tests.*",
            "*.tests",
            "*.tests.*"
        ]
    ),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.18",
        "matplotlib>=3.0",
        "scipy>=1.4",
        "seaborn>=0.10"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
