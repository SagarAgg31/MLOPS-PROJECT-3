from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requrirements = f.read().splitlines()

## ssss
setup(
    name='MLOPS-PROJECT-3',
    version='0.0.1',
    author = 'Sagar Aggarwal',
    packages= find_packages(),
    install_requires=requrirements,
)