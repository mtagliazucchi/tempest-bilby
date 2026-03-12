from setuptools import setup, find_packages

setup(
    name="tempest-bilby",
    version="0.1.0",
    description="Bilby plugin for the Tempest sampler",
    author="",
    packages=find_packages(),
    install_requires=[
        "bilby",
        "tempest-sampler",
        "numpy",
    ],
    entry_points={
        "bilby.samplers": [
            "tempest = tempest_bilby.tempest:Tempest"
        ]
    },
)
