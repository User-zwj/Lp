#!/usr/bin/env python

from distutils.core import setup

setup(name='lp',
      version='1.1',
      description='Lp convergence paper results',
      author='Wenjuan Zhang',
      author_email='wenjuan.zhang66@gmail.com',
      url='https://github.com/User-zwj/Lp',
      packages=['lp'],
      install_requires=['numpy', 'scipy',
                        'matplotlib', 'os',]
      )
