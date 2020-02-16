import cv2
import numpy as np
from cv2utils import FaceDnn

def main():
    
    image = cv2.imread("face.jpg")

    face_detector = FaceDnn()

    faces = face_detector.detect_faces(image)
    for face in faces:
        [x, y, x_final, y_final] = face['box']
        cv2.rectangle(image, (x,y), (x_final, y_final), (0, 0, 255), 2)
        text = "{:.2f}%".format(face['confidence'] * 100)
        cv2.putText(image, text, (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

    vis = np.concatenate((cv2.imread("face.jpg"), image), axis=1)
    cv2.imwrite("result_dnn.jpg", vis)
if __name__ == '__main__':
    main()