import cv2
import pytest
import numpy as np 
from cv2utils import EyeCascade

def test_detect_eyes():

    hodor = cv2.imread("hodor.jpg")
    eye_detector = EyeCascade()
    result = eye_detector.detect_eyes(hodor)

    assert len(result) == 2
    assert type(result[0]) is dict
    assert 'box' in result[0]
    assert 'label' in result[0]
    assert type(result[0]['box']) is list
    assert all([True if type(i) is int else False for i in result[0]['box']]) is True
    assert type(result[0]['label']) is str
    assert len(result[0]['box']) is 4
    

def test_invalid_image():
    not_image = cv2.imread("requirements-test.txt")
    eye_detector = EyeCascade()
    with pytest.raises(ValueError):
        result = eye_detector.detect_eyes(not_image)

def test_no_face():
    eye_detector = EyeCascade()
    no_eye = cv2.imread("no_face.jpg")
    result = eye_detector.detect_eyes(no_eye)
    
    assert len(result) == 0