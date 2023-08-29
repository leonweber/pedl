from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='pedl',
    version='1.0.2',
    description='Search the biomedical literature for protein interactions and'
                'protein associations.',
    url='https://github.com/leonweber/pedl',
    author='Leon Weber',
    author_email='leonweber@posteo.de',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["pedl-summarize=pedl.summarize:summarize",
                                      "pedl-rebuild_pubtator_index=pedl.rebuild_pubtator_index:rebuild_pubtator_index",
                                      "pedl-build_training_set=pedl.build_training_set:build_training_set",
                                      "pedl-extract=pedl.predict:predict"]},
    package_data={"pedl": ["data/*",
                           'configs/*',
                           'configs/database/*',
                           'configs/entities/*',
                           'configs/elastic/*',
                           'configs/type/*',
                           'configs/hydra/help/*']}
)
