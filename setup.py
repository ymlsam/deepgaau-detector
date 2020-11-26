from setuptools import setup, find_packages


setup(
    name='deepgaau-detector',
    version='0.0.0',
    packages=find_packages(include=[]),
    install_requires=[
        'labelImg==1.8.3',
        'matplotlib==3.3.2',
        'numpy==1.18.5',
        'Pillow==8.0.0',
        'scipy==1.5.4',
        'tensorflow==2.3.1',
        'tf-models-official==2.3.0',
        'tf-slim==1.1.0',
    ],
)