from setuptools import setup

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'deep_lincs',
    'version': '0.0.1',
    'description' : 'A framework for deep learning with LINCS L1000 Data',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
    'keywords' : 'LINCS L1000 gene expression',
    'url' : 'https://github.com/manzt/deep_lincs',
    'maintainer' : 'Trevor Manz',
    'maintainer_email' : 'trevor.j.manz@gmail.com',
    'license' : 'MIT',
    'packages' : ['deep_lincs'],
}

setup(**configuration)
