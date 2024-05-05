import cv2
from requests import post
import json
from json import JSONEncoder
import numpy as np

# capture image from mac os camera
cap = cv2.VideoCapture(1)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # connect to localhost:5000
    url = "http://127.0.0.1:5000/detect"

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    numpyData = {"image": frame}
    encoded_data = json.dumps(numpyData, cls=NumpyArrayEncoder)
    #response = post(
    #    url, data=encoded_data, headers={"Content-Type": "application/json"}
    #)
    #print(response.json())
