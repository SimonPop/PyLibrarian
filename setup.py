import os

from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pylibrarian",
    version="0.0.1",
    author="Simon Popelier",
    author_email="simon.popelier@gmail.com",
    description=("Recommender system for Python libraries."),
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Text Processing :: Linguistic",
    ],
    license="BSD",
    packages=find_packages(),
)
