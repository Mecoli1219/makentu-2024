import cv2
from requests import post
import json
from json import JSONEncoder
import numpy as np
import base64
import paho.mqtt.client as mqtt
import socket
import time
import io

NORMAL = 0
PLAYING = 1
CONTROL = 2

delay_time = 500

USE_PI_CAMERA = True

if not USE_PI_CAMERA:
    cap = cv2.VideoCapture(0)
else:
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.create_preview_configuration({"format": "RGB", "size": "640x480"})
    picam2.start()


class State:
    def __init__(self):
        self._state = NORMAL
        self.brick_num = 7
        self.brick_status = [[0, 0] for _ in range(self.brick_num)]
        self.brick_multiply = 100

    def set_state(self, state):
        if state in [NORMAL, PLAYING, CONTROL]:
            self._state = state

    def get_state(self):
        return self._state

    def brick_diff(self, brick_status):
        should_reset = True
        for i in range(self.brick_num):
            brick_status[i][0] = int(brick_status[i][0])
            brick_status[i][1] = int(brick_status[i][1])
            if brick_status[i][0] != 0 or brick_status[i][1] != 0:
                should_reset = False

        if should_reset:
            self.brick_status = brick_status
            return -1

        diff = [
            [
                self.brick_multiply * (brick_status[i][0] - self.brick_status[i][0]),
                self.brick_multiply * (brick_status[i][1] - self.brick_status[i][1]),
            ]
            for i in range(self.brick_num)
        ]
        self.brick_status = brick_status
        return diff

    def reset(self):
        self.brick_status = [[0, 0] for _ in range(self.brick_num)]
        return self.brick_status


STATE = State()


def get_frame():
    if USE_PI_CAMERA:
        data = io.BytesIO()
        picam2.capture_file(data, format="jpeg")
        data.seek(0)
        frame = np.frombuffer(data.getvalue(), dtype=np.uint8)
    else:
        ret, frame = cap.read()
    return frame


if __name__ == "__main__":

    HOST = "192.168.2.11"
    PORT = 6000
    SV_HOST = "10.20.2.41"
    SV_PORT = 8080

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", PORT))

    def send_motorCmd(motorCmd):
        motorCmd = STATE.brick_diff(motorCmd)
        if motorCmd == -1:
            data = "r"
        else:
            # ! Remove this
            motorCmd = motorCmd[0]
            data = json.dumps(motorCmd)

        s.sendto(data.encode(), (HOST, PORT))

    def on_message(client, obj, msg):
        print(f"TOPIC:{msg.topic}, VALUE:{msg.payload}")
        if msg.topic == "state":
            STATE.set_state(int(msg.payload))
            if STATE.get_state() == CONTROL:
                send_motorCmd(STATE.reset())
        if msg.topic == "control":
            if STATE.get_state() == CONTROL:
                data = msg.payload  # "[6, 1, 3]"
                motorCmd = json.loads(data)
                send_motorCmd(motorCmd)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(host=SV_HOST, port=1883)
    client.subscribe("state", 0)
    client.subscribe("control", 0)
    client.loop_start()

    url = "http://" + SV_HOST + ":" + str(SV_PORT) + "/"

    # capture image from mac os camera
    while True:
        frame = get_frame()
        if frame is None:
            print("Failed to get frame")
            continue

        base64img = base64.b64encode(frame)
        encoded_data = json.dumps({"image": base64img.decode("utf-8")})
        if STATE.get_state() == NORMAL:
            response = post(
                url + "detect",
                data=encoded_data,
                headers={"Content-Type": "application/json"},
            )
            if response.ok:
                motorCmd = response.json()
                send_motorCmd(motorCmd)
                time.sleep(delay_time / 1000)
            else:
                print("Failed to get response")
                time.sleep(delay_time / 10000)
        elif STATE.get_state() == PLAYING:
            response = post(
                url + "image",
                data=encoded_data,
                headers={"Content-Type": "application/json"},
            )

            response = post(url + "playing")
            motorCmd = response.json()
            send_motorCmd(motorCmd)
            time.sleep(delay_time / 1000)
        else:
            # print("Control state")
            response = post(
                url + "image",
                data=encoded_data,
                headers={"Content-Type": "application/json"},
            )
            time.sleep(delay_time / 1000)

    s.close()
