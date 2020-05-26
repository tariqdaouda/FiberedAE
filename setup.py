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

    author='Tariq Daouda, Reda Chhaibi, Prudencio Tossou',
    author_email='',

    license='MIT',
    
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    python_requires='>=3.5',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],

    install_requires=["numpy", "scipy", "matplotlib", "holoviews", "torch", "umap-learn", "scanpy", "scgen", "click", "pandas", "scikit-image", "torchvision", "tqdm", "harmonypy"],

    keywords='',

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            ['fae = fiberedae.__main__:main'],
            ['fae-translate-sc = fiberedae.__main__:translate']
        ],
    },
    package_dir={'fiberedae': 'fiberedae'}
    # packages=['fiberedae']
)
