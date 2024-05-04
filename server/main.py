from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2
import json

import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt

from utils.decisionMaker import DecisionMaker
from utils.constant import PLAYING

app = Flask(__name__)
decisionMaker = DecisionMaker(model_path="./utils/pose_landmarker.task")
app.config["SECRET_KEY"] = "MaKeNTu2024grOuP8"

client = mqtt.Client()
client.connect(host="127.0.0.1", port=1883)
client.loop_start()


@app.route("/detect", methods=["POST"])
def detect():
    if decisionMaker.getState() == PLAYING:
        client.publish(topic="command", payload=str(PLAYING))
        return jsonify({"error": "It is playing State"}), 400
    encoded_data = request.data
    img = json.loads(encoded_data)["image"]
    img = np.array(img, dtype=np.uint8)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    try:
        result = decisionMaker.makeDecision(img.numpy_view())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # if debug save img
    if app.debug:
        cv2.imwrite("./inputs/img.png", img)
        block_img = decisionMaker.plotBrickStatus(img)
        annotated_img = decisionMaker.detector.result_image(
            block_img, decisionMaker.detection_result
        )
        cv2.imwrite("./outputs/img.png", annotated_img)
    if result is None:
        return jsonify({"error": "No result"}), 400
    return jsonify(result)


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
    x, y, level = control
    result = decisionMaker.updateControl(x, y, level)
    if result is None:
        return jsonify({"error": "Invalid control"}), 400
    payload = json.dumps(control)
    client.publish(topic="control", payload=payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
