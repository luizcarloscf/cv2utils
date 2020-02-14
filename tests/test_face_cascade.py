import cv2
import pytest
import numpy as np 
from cv2utils import FaceCascade

def test_detect_faces():

    hodor = cv2.imread("hodor.jpg")
    face_detector = FaceCascade()
    result = face_detector.detect_faces(hodor)

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
    face_detector = FaceCascade()
    with pytest.raises(ValueError):
        result = face_detector.detect_faces(not_image)

def test_no_face():
    face_detector = FaceCascade()
    no_face = cv2.imread("no_face.jpg")
    result = face_detector.detect_faces(no_face)
    
    assert len(result) == 0