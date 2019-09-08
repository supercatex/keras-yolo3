import cv2


model = cv2.dnn.readNet("model_data/my_yolo.pb")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break
camera.release()
cv2.destroyAllWindows()
