from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="natural_galore",
    version="0.1",
    description="Natural GaLore",
    url="",
    author="Anonymous",
    author_email="Anonymous",
    license="Apache 2.0",
    packages=["natural_galore"],
    install_requires=required,
)
