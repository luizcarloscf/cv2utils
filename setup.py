from os import path
from setuptools import setup, find_packages

__author__ = "Luiz Carlos Cosmi Filho"
__version__= "0.0.8"


def readme():
    directory = path.abspath(path.dirname(__file__))
    with open(path.join(directory, 'README.md'), encoding="UTF-8") as f:
        return f.read()

setup(
    name='cv2utils',
    version='0.0.8',
    description='Implementation of some object detection',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/luizcarloscf/cv2utils.git',
    author='Luiz Carlos Cosmi Filho',
    author_email='luizcarloscosmifilho@gmail.com',
    license='MIT',
    packages=['cv2utils'],
    zip_safe=False,
    install_requires=[
        'opencv-contrib-python==4.2.0.32',
        'numpy==1.18.1',
    ],
    classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
    ],
    keywords="dnn haar cascade face eye detection opencv numpy pip package",
    include_package_data=True,
)
