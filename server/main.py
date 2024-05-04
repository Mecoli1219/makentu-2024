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
decisionMaker = DecisionMaker(num_brick=7, model_path="./utils/pose_landmarker.task")
app.config["SECRET_KEY"] = "MaKeNTu2024grOuP8"

client = mqtt.Client()
client.connect(host="127.0.0.1", port=1883)
client.loop_start()


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/detect", methods=["POST"])
def detect():
    print("Receive Detect")
    if decisionMaker.getState() == PLAYING:
        client.publish(topic="command", payload=str(PLAYING))
        return jsonify({"error": "It is playing State"}), 400
    encoded_data = request.data

    img = json.loads(encoded_data)["image"]
    img = base64.b64decode(img.encode("utf-8"))
    img = np.frombuffer(img, dtype=np.uint8)
    img = img.reshape((480, 640, 3))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    client.publish(topic="image", payload=encoded_data)

    try:
        result = decisionMaker.makeDecision(mp_img.numpy_view())
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if result is None:
        return jsonify({"error": "No result"}), 400

    # if app.debug:
    cv2.imwrite("./inputs/img.png", img)
    block_img = decisionMaker.plotBrickStatus(img)
    annotated_img = decisionMaker.detector.result_image(
        block_img, decisionMaker.detection_result
    )
    ws_img = Image.fromarray(annotated_img)
    byte_array = pil_image_to_byte_array(ws_img)
    client.publish(topic="debug-image", payload=byte_array)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./outputs/img.png", annotated_img)

    return jsonify(result)


@app.route("/image", methods=["POST"])
def image():
    encoded_data = request.data
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
