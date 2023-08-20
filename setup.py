from setuptools import setup, find_packages

setup(
    name='exllama',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'safetensors==0.3.1',
        'sentencepiece>=0.1.97',
        'ninja==1.11.1',
        'flask~=2.3.2',
        'waitress~=2.1.2'
    ],
)
