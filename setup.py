from setuptools import setup, find_packages

setup(
    name="pynav",
    version="0.0.1",
    description="A collection of common state estimation algorithms in robotics.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.2",
        "scipy>=1.7.1",
        "matplotlib>=3.4.3",
        "joblib>=1.2.0",
        'pylie @ git+https://github.com/decargroup/pylie@main'
    ]
)
