##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

from setuptools import setup, find_packages

setup(name='text2pose',
      version='1.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      author='Ginger Delmas',
      author_email='ginger.delmas.pro@gmail.com',
      description='PoseScript: 3D Human Poses from Natural Language; text-conditioned 3D human pose retrieval and generation.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      install_requires=[],
      dependency_links=[],
      )