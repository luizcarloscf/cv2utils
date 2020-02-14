from os import path
from setuptools import setup, find_packages

__author__ = "Luiz Carlos Cosmi Filho"
__version__= "0.0.7"


def readme():
    directory = path.abspath(path.dirname(__file__))
    with open(path.join(directory, 'README.md'), encoding="UTF-8") as f:
        return f.read()

setup(
    name='cv2utils',
    version='0.0.7',
    description='Face detector wrapper',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/luizcarloscf/cv2utils.git',
    author='Luiz Carlos Cosmi Filho',
    author_email='luizcarloscosmifilho@gmail.com',
    license='MIT',
    packages=['cv2utils'],
    zip_safe=False,
    install_requires=[
        'opencv-python==4.1.2.*',
        'numpy==1.16.1',
    ],
    include_package_data=True,
)
