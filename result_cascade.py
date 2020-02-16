import cv2
import numpy as np
from cv2utils import FaceCascade, EyeCascade, FaceDnn

def main():
    
    image = cv2.imread("face.jpg")

    face_detector = FaceCascade()
    eye_detector = EyeCascade()

    faces = face_detector.detect_faces(image)
    for face in faces:
        [x, y, x_final, y_final] = face['box']
        cv2.rectangle(image, (x,y), (x_final, y_final), (255, 0, 255), 2)
        
        roi = image[y:y_final, x:x_final]
        eyes = eye_detector.detect_eyes(roi)
        for eye in eyes:
            [x, y, x_final, y_final] = eye['box']
            cv2.rectangle(roi,(x,y),(x_final,y_final),(0,0,255),2)

    vis = np.concatenate((cv2.imread("face.jpg"), image), axis=1)
    cv2.imwrite("result_cascade.jpg", vis)
if __name__ == '__main__':
    main()