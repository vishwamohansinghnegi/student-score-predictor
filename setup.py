from setuptools import  find_packages , setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path)->List[str]:      # Output of fx will be a list
    ''' This fx will return list of requirements'''
    requirements = list()
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n' , '') for req in requirements]  # will remove '\n' as lib are written in different lines
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name = 'student-score-predictor',
    version = '0.0.1',
    author = 'Vishwamohan Singh Negi',
    author_email = 'vishwamohansinghnegi@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)