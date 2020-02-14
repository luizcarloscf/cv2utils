import os
import cv2
import numpy as np
from pkg_resources import Requirement, resource_filename


class FaceCascade(object):

    def __init__(self,
                 weights_file: str = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: tuple = (10, 10)):

        if weights_file is None:
            weights_file = resource_filename(
                Requirement.parse('cv2utils'), 'cv2utils' + os.path.sep + 'data' + os.path.sep +
                'haarcascade_frontalface_default.xml')

        if type(scale_factor) is not float and type(scale_factor) is not int:
            raise ValueError("scale_factor must be a float or int")

        if type(min_neighbors) is not int:
            raise ValueError("min_neighbors must be a integer")

        if len(min_size) > 2:
            raise ValueError("min_size must be a tuple of two integers values")

        if any([True if type(i) is not int else False for i in min_size]):
            raise ValueError("min_size must be a tuple of two integers values")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.face_cascade = cv2.CascadeClassifier(weights_file)

    def detect_faces(self, image):

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        faces = self.face_cascade.detectMultiScale(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)

        return [{'label': 'face', 'box': face.tolist()} for face in faces]


class EyeCascade(object):

    def __init__(self,
                 weights_file: str = None,
                 scale_factor: float = 1.3,
                 min_neighbors: int = 6,
                 min_size: tuple = (30, 30)):

        if weights_file is None:
            weights_file = resource_filename(
                Requirement.parse('cv2utils'), 'cv2utils' + os.path.sep + 'data' + os.path.sep +
                'haarcascade_eye.xml')

        if type(scale_factor) is not float and type(scale_factor) is not int:
            raise ValueError("scale_factor must be a float or int")

        if type(min_neighbors) is not int:
            raise ValueError("min_neighbors must be a integer")

        if len(min_size) > 2:
            raise ValueError("min_size must be a tuple of two integers values")

        if any([True if type(i) is not int else False for i in min_size]):
            raise ValueError("min_size must be a tuple of two integers values")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        print(weights_file)
        self.eye_cascade = cv2.CascadeClassifier(weights_file)

    def detect_eyes(self, image):

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        eyes = self.eye_cascade.detectMultiScale(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)

        return [{'label': 'eye', 'box': eye.tolist()} for eye in eyes]