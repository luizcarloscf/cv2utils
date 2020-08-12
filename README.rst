========
cv2utils
========

.. image:: https://img.shields.io/pypi/v/cv2utils.svg?label=cv2utils
    :target: https://pypi.org/project/cv2utils
    :alt: PyPI 

.. image:: http://img.shields.io/travis/luizcarloscf/cv2utils/master.svg?label=linux
    :target: https://travis-ci.com/luizcarloscf/cv2utils
    :alt: Travis

.. image:: https://readthedocs.org/projects/cv2utils/badge/?version=latest
    :target: https://cv2utils.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/dm/cv2utils
    :target: https://pypi.org/project/cv2utils
    :alt: PyPI - Downloads

.. image:: https://img.shields.io/badge/license-MIT%20-blue.svg
    :target: https://github.com/luizcarloscf/cv2utils/LICENSE




Implementation of some object detection in Python3.5+. It is included in this project:

* Face and Eye detection using OpenCV Haar feature-based cascade classifiers.

.. image:: https://raw.githubusercontent.com/luizcarloscf/cv2utils/master/examples/images/result_cascade.jpg
    :align: center
    :target: https://github.com/luizcarloscf/cv2utils/blob/master/examples/images/result_cascade.jpg
    :alt: Result Cascade


* Face detection using OpenCV Deep Neural Networks

.. image:: https://raw.githubusercontent.com/luizcarloscf/cv2utils/master/examples/images/result_dnn.jpg
    :align: center
    :target: https://github.com/luizcarloscf/cv2utils/blob/master/examples/images/result_dnn.jpg
    :alt: Result DNN


For for info, `Read the docs <https://cv2utils.readthedocs.io/en/latest/>`__.

Installation
------------

It can be installed through pip:

.. code-block:: bash

    pip3 install --user cv2utils

This implementation requires OpenCV and Numpy.

References
----------

* `OpenCV HaarCascades <https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html>`__

* `OpenCV Dnn <https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html>`__

License
-------

`MIT <https://github.com/luizcarloscf/cv2utils/blob/master/LICENSE>`__