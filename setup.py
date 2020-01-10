from setuptools import setup, find_packages

setup(
    name='cvutils',
    version='0.0.1',
    description='Face detector wrapper',
    long_description='Face detector wrapper',
    url='https://github.com/luizcarloscf/cvutils.git',
    author='Luiz Carlos Cosmi Filho',
    author_email='luizcarloscosmifilho@gmail.com',
    license='MIT',
    packages=find_packages('cvutils'),
    package_dir={'': 'cvutils'},
    zip_safe=False,
    install_requires=[
        'opencv-python=>4.1.0',
        'numpy=>1.16.1',
    ],
)
