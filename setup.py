from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='pedl',
    version='0.1.0',
    description='Search the biomedical literature for protein interactions and'
                'protein associations.',
    url='https://github.com/leonweber/pedl',
    author='Leon Weber',
    author_email='leonweber@posteo.de',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["pedl=pedl.predict:main"]}
)