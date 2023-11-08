##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

from setuptools import setup, find_packages

setup(name='text2pose',
      version='2.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      author='Ginger Delmas',
      author_email='ginger.delmas.pro@gmail.com',
      description='PoseScript ECCV22 & PoseFix ICCV23.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      )