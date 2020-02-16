import cv2
import pytest
import numpy as np 
from cv2utils import EyeCascade, FaceCascade

def test_detect_eyes():

    image = cv2.imread("face.jpg")
    eye_detector = EyeCascade()
    face_detector = FaceCascade()

    faces = face_detector.detect_faces(image)
    face = faces[0]['box']

    (x,y,w,h) = face
    roi = image[y:y+h, x:x+w]
    result = eye_detector.detect_eyes(roi)
        

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

def test_no_eye():
    eye_detector = EyeCascade()
    no_eye = cv2.imread("no_face.jpg")
    result = eye_detector.detect_eyes(no_eye)
    
    assert len(result) == 0