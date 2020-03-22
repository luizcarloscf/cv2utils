import os
import cv2
from pkg_resources import Requirement, resource_filename


class FaceCascade(object):
    """Face detection using Haar feature-based cascade classifiers.

    Parameters
    ---------- 

    weights_file: str, optional
        Path to the xml file.
    
    scale_factor: float, optional
        Scaling by some factor (create a scale pyramid)

    min_neighbots: int, optional
        How many neighbors each candidate rectangle should have to retain it        

    size: tuple, optional
        Minimal size

    Examples
    --------
    >>> import cv2
    >>> from cv2utils import FaceCascade
    >>> face_detector = FaceCascade()

    """
    def __init__(self,
                 weights_file: str = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: tuple = (10, 10)):

        if weights_file is None:
            weights_file = resource_filename(
                Requirement.parse('cv2utils'),
                'cv2utils' + os.path.sep + 'data' + os.path.sep + 'haarcascade_frontalface_default.xml')

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
        """Apply the image on classifier.

        Parameters
        ----------
        
        image: numpy.array
            Image (BGR color space) for detection face on it.
        
        Returns
        -------
        list
            a list of all dicts containg all detections. Each dict contains: box (The bounding box
            is formatted as [x_initial, y_initial, x_final, y_final] under the key 'box') 
            and label (identifies which object is detecting)

        Examples
        --------
        >>> import cv2
        >>> from cv2utils import FaceCascade, EyeCascade
        >>> image = imread("face.jpg")
        >>> face_detector = FaceCascade()
        >>> faces = face_detector.detect_faces(image)
        >>> faces
        [{'label': 'face', 'box': [199, 65, 591, 457]}]
        """

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        faces = self.face_cascade.detectMultiScale(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)

        boxes = list()
        for face in faces:
            [x, y, w, h] = face.tolist()
            boxes.append([x, y, x + w, y + h])

        return [{'label': 'face', 'box': box} for box in boxes]


class EyeCascade(object):
    """Eye detection using Haar feature-based cascade classifiers.

    Parameters
    ---------- 

    weights_file: str, optional
        Path to the xml file.
    
    scale_factor: float, optional
        Scaling by some factor (create a scale pyramid)

    min_neighbots: int, optional
        How many neighbors each candidate rectangle should have to retain it        

    size: tuple, optional
        Minimal size

    Examples
    --------
    >>> import cv2
    >>> from cv2utils import EyeCascade
    >>> eye_detector = EyeCascade()
    """
    def __init__(self,
                 weights_file: str = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 3,
                 min_size: tuple = (0, 0)):

        if weights_file is None:
            weights_file = resource_filename(
                Requirement.parse('cv2utils'),
                'cv2utils' + os.path.sep + 'data' + os.path.sep + 'haarcascade_eye.xml')

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
        self.eye_cascade = cv2.CascadeClassifier(weights_file)

    def detect_eyes(self, image):
        """Apply the image on classifier.

        Parameters
        ----------
        
        image: numpy.array
            Image (BGR color space) for detection eye on it.
        
        Returns
        -------
        list
            a list of all dicts containg all detections. Each dict contains: box (The bounding box
            is formatted as [x_initial, y_initial, x_final, y_final] under the key 'box') 
            and label (identifies which object is detecting)

        Examples
        --------

        This classifier was trained with images with low resolution. So, for better results, apply
        the face detector first and create a Region Of Interest (ROI).
        
        >>> import cv2
        >>> from cv2utils import FaceCascade, EyeCascade
        >>> image = imread("face.jpg")
        >>> face_detector = FaceCascade()
        >>> faces = face_detector.detect_faces(image)
        >>> faces
        [{'label': 'face', 'box': [199, 65, 591, 457]}]
        >>>
        >>> [x,y,x_final,y_final] = faces[0]['box']
        >>> eye_detector = EyeCascade()
        >>> eye_detector.detect_eyes(image[y:y_final, x:x_final])
        [{'label': 'eye', 'box': [83, 132, 166, 215]}, {'label': 'eye', 'box': [218, 119, 298, 199]}]
        """

        if image is None or not hasattr(image, "shape"):
            raise ValueError("Image not valid.")

        eyes = self.eye_cascade.detectMultiScale(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size)

        boxes = list()
        for eye in eyes:
            [x, y, w, h] = eye.tolist()
            boxes.append([x, y, x + w, y + h])

        return [{'label': 'eye', 'box': box} for box in boxes]
