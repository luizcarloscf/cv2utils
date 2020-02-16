import cv2
import pytest
import numpy as np 
from cv2utils import FaceDnn

def test_detect_faces():

    face = cv2.imread("face.jpg")
    face_detector = FaceDnn()
    result = face_detector.detect_faces(face)

    assert len(result) == 1
    assert type(result[0]) is dict
    assert 'box' in result[0]
    assert 'label' in result[0]
    assert 'confidence' in result[0]
    assert type(result[0]['box']) is list
    assert type(result[0]['label']) is str
    assert type(result[0]['confidence']) is float
    assert all([True if type(i) is int else False for i in result[0]['box']]) is True
    assert len(result[0]['box']) is 4
    

def test_invalid_image():
    not_image = cv2.imread("requirements-test.txt")
    face_detector = FaceDnn()
    with pytest.raises(ValueError):
        result = face_detector.detect_faces(not_image)

def test_no_face():
    face_detector = FaceDnn()
    no_face = cv2.imread("no_face.jpg")
    result = face_detector.detect_faces(no_face)
    
    assert len(result) == 0