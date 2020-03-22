import os
import cv2
import numpy as np
from pkg_resources import Requirement, resource_filename


class FaceDnn(object):
    """Implemented Face Detection using Deep Neural Network implemented in OpenCV. 
        
        OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework
        with a ResNet base network.

        Parameters
        ----------
        
        prototxt_file: str, optional
            Path to the Proto Caffe file.
        
        caffemodel_file: str, optional
            Path to the pretrained Caffe model.
        
        threshold: float, optional
            The model returns the confidence, by default filter detections with 50% confidence.

        scale_factor: float, optional
            Scaling by some factor

        size: tuple, optional
            Spatial size that the Neural Network expects.

        mean: tuple, optional
            Mean subtraction.

        Examples
        --------

        >>> import cv2
        >>> from cv2utils import FaceDnn
        >>> detector = FaceDnn()
    """

    def __init__(self,
                 prototxt_file: str = None,
                 caffemodel_file: str = None,
                 threshold: float = 0.5,
                 scale_factor: float = 1.0,
                 size: tuple = (300, 300),
                 mean: tuple = (104.0, 177.0, 123.0)):
        if prototxt_file is None:
            prototxt_file = resource_filename(
                Requirement.parse('cv2utils'),
                'cv2utils' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt.txt')

        if caffemodel_file is None:
            caffemodel_file = resource_filename(
                Requirement.parse('cv2utils'),
                'cv2utils' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        
        self.network = cv2.dnn.readNetFromCaffe(prototxt_file, caffemodel_file)
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.size = size
        self.mean = mean

    def detect_faces(self, image):
        """Converts the image to blob and apply on the model.

        Parameters
        ----------

        image: numpy.array
            Image (BGR color space) for detection face on it.
        
        Returns
        -------
        list
            a list of all dicts containg all detections. Each dict contains: box (The bounding box
            is formatted as [x_initial, y_initial, x_final, y_final] under the key 'box'), 
            confidence (is the probability estimate for a bounding box to be matching the label) 
            and label (identifies which object is detecting)
        
        Examples
        --------
        >>> import cv2
        >>> from cv2utils import FaceDnn
        >>> detector = FaceDnn()
        >>> image = cv2.imread("face.jpg")
        >>> detector.detect_faces(image)
        [{'label': 'face', 'confidence': 0.9966524243354797, 'box': [210, 64, 522, 465]}]
        """

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
