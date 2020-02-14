import os
import cv2
import numpy as np
from pkg_resources import Requirement, resource_filename


class FaceDnn(object):

    def __init__(self,
                 prototxt_file: str = None,
                 caffemodel_file: str = None,
                 threshold: float = 0.5,
                 scale_factor: float = 1.0,
                 size: tuple = (300, 300),
                 mean: tuple = (104.0, 177.0, 123.0)):
        if prototxt_file is None:
            prototxt = resource_filename(
                Requirement.parse('cv2utils'),
                'cv2utils' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt.txt')
        else:
            prototxt = prototxt_file

        if caffemodel_file is None:
            caffemodel = resource_filename(
                Requirement.parse('cv2utils'), 'cv2utils' + os.path.sep + 'data' + os.path.sep +
                'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        else:
            caffemodel = caffemodel_file

        self.network = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.size = size
        self.mean = mean

    def detect_faces(self, image):

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        all_detections = []

        origin_h, origin_w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, self.size), self.scale_factor, self.size, self.mean)
        self.network.setInput(blob)

        detections = self.network.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > self.threshold:

                bounding_box = detections[0, 0, i, 3:7] * np.array(
                    [origin_w, origin_h, origin_w, origin_h])
                x_start, y_start, x_end, y_end = bounding_box.astype('int')

                all_detections.append({
                    'label': 'face',
                    'confidence': float(confidence),
                    'box': [int(x_start), int(y_start), int(x_end), int(y_end)]
                })

        return all_detections
