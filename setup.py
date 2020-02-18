from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='fiberedae',

    version='0.01',

    description='Implementation of Fibered Auto Encoders (FAE)',
    long_description=""" Implementation of Fibered Auto Encoders (FAE).
    FAE is a short name which refers to Auto Encoders whose space of latent variables have a fiber bundle structure. 
    This implementation acts as a proof of concept and aims at illustrating the concept""",
    url='',

    author='Anonymous Authors',
    author_email='',

    license='ApacheV2',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],

    install_requires=["numpy", "scipy", "matplotlib", "holoviews", "torch", "umap-learn", "scanpy", "scgen", "click", "pandas", "scikit-image", "torchvision", "tqdm", "harmonypy"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'sample=sample:main',
        ],
    },
)
