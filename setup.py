#!/usr/bin/env python
from setuptools import setup

setup(name='cvhci',
      version='0.0.1',
      install_requires=[
          'scikit-image', 'torch', 'opencv-python', 'tensorboardX', 'torchvision'
      ]
      )
