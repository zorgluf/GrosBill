from setuptools import setup, find_packages

setup(
    name='frouge',
    version='0.1.0',
    description='Flamme Rouge Gym Environment',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'opencv-python',
        'shimmy',
        'sb3-contrib',
        'nicegui'
    ]
)


