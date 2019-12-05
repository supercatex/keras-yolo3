from flask import Flask, request, Response
import cv2
import numpy as np
import jsonpickle


app = Flask(__name__)


@app.route("/")
def index():
    return "My yolo Service"


@app.route("/detect", methods=["POST"])
def detect():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    response = {"message": "image received. size={}x{}".format(img.shape[1], img.shape[0])}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)
