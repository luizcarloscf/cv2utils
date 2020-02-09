import cv2
import pytest
import numpy as np 
from cv2utils import FaceCascade

def test_detect_faces():

    hodor = cv2.imread("hodor.jpg")
    face_detector = FaceCascade()
    result = face_detector.detect_faces(hodor)

    assert len(result) == 1

    first = result[0]

    assert type(first) is dict
    assert 'box' in first
    assert type(first['box']) is list
    

    