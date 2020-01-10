import cv2
import numpy as np
import pkg_resources


class HaarCascadeFace(object):

    def __init__(self,
                 weights_file: str = None,
                 scaleFactor: float = 1.1,
                 minNeighbors: int = 5,
                 minSize: tuple = (1, 1)):

        if weights_file is None:
            weights_file = pkg_resources.resource_stream(
                'cvutils', 'data/haarcascade_frontalface_default.xml')

        if len(minSize) > 2:
            raise ValueError("Must be a tuple of two integers values")

        if all([True if type(i) is int else False for i in minSize]):
            raise ValueError("Must be a tuple of two integers values")

        self.scale_factor = scaleFactor
        self.min_neighbors = minNeighbors
        self.min_size = minSize
        self.faceCascade = cv2.CascadeClassifier(weights_file)

    def detect_face(self, image):

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        faces = self.faceCascade.detectMultiScale(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)

        return faces
