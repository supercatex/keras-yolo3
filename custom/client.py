import requests
import json
import cv2


server = "http://localhost:5000"
service = server + "/detect"

content_type = "image/jpeg"
headers = {"content-type": content_type}

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(service, data=img_encoded.tostring(), headers=headers)
    print(json.loads(response.text))

    cv2.waitKey(0)
