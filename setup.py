from setuptools import setup,find_packages

with open("requirements.txt") as file:
    requirements=file.read().splitlines()


setup(
    name="Mlops-CI-CD",
    version="0.1",
    author= "Taufeeq",
    packages= find_packages(),
    install_requires= requirements,
)