from setuptools import setup, find_packages

setup(
    name='CPGEO',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.0',
    ],
    package_data={
        'CPGEO': ['dlls/*'],
        'CPGEO.utils': ['dlls/*'],
    },
    author='Song Zenan',
    include_package_data=True,
)