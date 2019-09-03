import cv2
import numpy as np
from PIL import Image
from yolo import YOLO


model = YOLO(model_path="model_data/my_yolo.h5", classes_path="model_data/voc_classes.txt")

cam = cv2.VideoCapture(0)
while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break

    img = Image.fromarray(frame)
    img = model.detect_image(img)
    img = np.asarray(img)

    cv2.imshow("frame", frame)
    cv2.imshow("image", img)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break
cam.release()
cv2.destroyAllWindows()
