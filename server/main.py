from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2
import json

import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
from PIL import Image
from io import BytesIO
import base64

from utils.decisionMaker import DecisionMaker
from utils.constant import PLAYING
from utils.helper import pil_image_to_byte_array

app = Flask(__name__)
app.config["SECRET_KEY"] = "MaKeNTu2024grOuP8"

client = mqtt.Client()
client.connect(host="127.0.0.1", port=1883)
client.loop_start()

USE_CAMERA = True
if USE_CAMERA:
    cap = cv2.VideoCapture(1)
decisionMaker = DecisionMaker(num_brick=7, model_path="./utils/pose_landmarker.task")


@app.route("/")
def index():
    return "Hello, World!"


def preprocess_image(img):
    frame = img[300:1500, 600:1280, :]
    return frame


@app.route("/detect", methods=["POST"])
def detect():
    if decisionMaker.getState() == PLAYING:
        client.publish(topic="command", payload=str(PLAYING))
        print("It is playing State")
        return jsonify({"error": "It is playing State"}), 400
    if not USE_CAMERA:
        encoded_data = request.data

        img = json.loads(encoded_data)["image"]
        img = base64.b64decode(img.encode("utf-8"))
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((480, 640, 3))
    else:
        ret, img = cap.read()
        # resize image
        if not ret:
            print("Failed to get frame")
            return jsonify({"error": "Failed to get frame"}), 400
        img = preprocess_image(img)
        file_img = np.ascontiguousarray(img)
        encoded_data = json.dumps({"image": base64.b64encode(file_img).decode("utf-8")})

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=file_img)

    client.publish(topic="image", payload=encoded_data)

    # try:
    result = decisionMaker.makeDecision(mp_img.numpy_view())
    # except Exception as e:
    #     print(e)
    #     return jsonify({"error": str(e)}), 400
    if result is None:
        print("No result")
        return jsonify({"error": "No result"}), 400

    # if app.debug:
    cv2.imwrite("./inputs/img.png", img)
    block_img = decisionMaker.plotBrickStatus(img)
    annotated_img = decisionMaker.detector.result_image(
        block_img, decisionMaker.detection_result
    )
    encoded_data = json.dumps(
        {"image": base64.b64encode(annotated_img).decode("utf-8")}
    )
    client.publish(topic="debug-image", payload=encoded_data)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./outputs/img.png", annotated_img)

    return jsonify(result)


@app.route("/image", methods=["POST"])
def image():
    if not USE_CAMERA:
        encoded_data = request.data
    else:
        ret, img = cap.read()
        if not ret:
            return jsonify({"error": "Failed to get frame"}), 400
        img = preprocess_image(img)
        encoded_data = json.dumps({"image": base64.b64encode(img).decode("utf-8")})
    client.publish(topic="image", payload=encoded_data)
    return jsonify({"message": "Image received"})


@app.route("/playing", methods=["POST"])
def playing():
    result = decisionMaker.updatePlaying()
    return jsonify(result)


@app.route("/state", methods=["POST"])
def state():
    state = request.data
    result = decisionMaker.updateState(int(state))
    client.publish(topic="state", payload=state)
    return jsonify(result)


@app.route("/control", methods=["POST"])
def control():
    control = json.loads(request.data)
    if len(control) != decisionMaker.num_brick:
        return jsonify({"error": "Invalid control"}), 400
    for c in range(len(control)):
        if len(control[c]) != 2:
            return jsonify({"error": "Invalid control"}), 400
        if (
            control[c][0] < 0
            or control[c][0] > decisionMaker.max_level
            or control[c][1] < 0
            or control[c][1] > decisionMaker.max_level
        ):
            return jsonify({"error": "Invalid control"}), 400
        control[c][0] = int(control[c][0])
        control[c][1] = int(control[c][1])
    result = decisionMaker.updateControl(control)

    if result is None:
        return jsonify({"error": "Invalid control"}), 400
    payload = json.dumps(control)
    client.publish(topic="control", payload=payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
