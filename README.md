# cv2utils

![PyPI](https://img.shields.io/pypi/v/cv2utils.svg?label=cv2utils)
![Travis](https://img.shields.io/travis/com/luizcarloscf/cv2utils.svg?label=Linux)

Implementation of some object detection in Python3.5+.

## Installation

It can be installed through pip:
```bash
pip3 install --user cv2utils
```
This implementation requires OpenCV and Numpy.

## Usage

#### OpenCV Face DNN

The following example illustrates the ease of use of this package:
```python
>>> import cv2
>>> from cv2utils import FaceDnn
>>> image = cv2.imread("face.jpg")
>>> detector = FaceDnn()
>>> detector.detect_faces(image)
[{'label': 'face', 'confidence': 0.9966524243354797, 'box': [210, 64, 522, 465]}]
```

The detector returns a list of DICTIONARY objects. Each DICTIONARY object contains three main keys: 'box', 'confidence' and 'label':

* The bounding **box** is formatted as [x_initial, y_initial, x_final, y_final] under the key 'box'.
* The **confidence** is the probability estimate for a bounding box to be matching the label.
* The **label** identifies which object is detecting.

Look the file [result_dnn.py](https://github.com/luizcarloscf/cv2utils/blob/master/result_dnn.py) to see how the image below was generated.

<p align="center"><img src="https://raw.githubusercontent.com/luizcarloscf/cv2utils/master/result_dnn.jpg" align=middle width=700pt height=250pt/></p>


#### OpenCV Face Cascade and Eye Cascade

The following example illustrates the ease of use of this package:

```python
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
```

The detector returns a list of DICTIONARY objects. Each DICTIONARY object contains two main keys: 'box', 'label':

* The bounding **box** is formatted as [x_initial, y_initial, x_final, y_final] under the key 'box'.
* The **label** identifies which object is detecting.

Look the file [result_cascade.py](https://github.com/luizcarloscf/cv2utils/blob/master/result_cascade.py) to see how the image below was generated.

<p align="center"><img src="https://raw.githubusercontent.com/luizcarloscf/cv2utils/master/result_cascade.jpg" align=middle width=700pt height=250pt/></p>


## References

* [OpenCV HaarCascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

* [OpenCV Dnn](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)

## License

[MIT](https://github.com/luizcarloscf/cv2utils/blob/master/LICENSE).
