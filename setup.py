from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

about = {}
with open(path.join(here, 'cv2utils/__version__.py'), encoding='UTF-8') as f:
    exec(f.read(), about)


with open(path.join(here, 'README.rst'), encoding="UTF-8") as f:
    readme = f.read()

setup(
    name=about['__package__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/x-rst',
    url=about['__url__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    packages=[about['__package__']],
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
