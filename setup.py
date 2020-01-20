from setuptools import setup, find_packages

setup(
    name='cv2utils',
    version='0.0.3',
    description='Face detector wrapper',
    long_description='Face detector wrapper',
    url='https://github.com/luizcarloscf/cv2utils.git',
    author='Luiz Carlos Cosmi Filho',
    author_email='luizcarloscosmifilho@gmail.com',
    license='MIT',
    packages=find_packages('cv2utils'),
    package_dir={'': 'cv2utils'},
    zip_safe=False,
    install_requires=[
        'opencv-python==4.1.2.*',
        'numpy==1.16.1',
    ],
    include_package_data=True,
)
