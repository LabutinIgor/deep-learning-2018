from setuptools import setup

setup(name='resnext',
      version='0.1',
      description='Resnext implementation based on pytorch resnet',
      author='Labutin Igor',
      packages=['resnext', 'test'],
      requires=['torch', 'torchvision', 'tensorboardX', 'pytest']
      )
