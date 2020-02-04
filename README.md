```python3
>>> import cv2
>>> from cv2utils import HaarCascadeFace
>>> img = cv2.imread("hodor.jpg")
>>> detector = HaarCascadeFace()
>>> detector.detect_face(img)
array([[355,  41, 136, 136],
       [909, 334,  51,  51]], dtype=int32)

```