import cv2
from requests import post
import json
import base64
import paho.mqtt.client as mqtt
import socket
import time

NORMAL = 0
PLAYING = 1
CONTROL = 2

delay_time = 500

USE_CAMERA = False
DEBUG = False

if USE_CAMERA:
    cap = cv2.VideoCapture(0)


class State:
    def __init__(self):
        self._state = NORMAL
        self.brick_num = 7
        self.brick_status = [[0, 0] for _ in range(self.brick_num)]
        self.brick_multiply = 300

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
    ret, frame = cap.read()
    return frame


if __name__ == "__main__":

    SV_HOST = "10.20.2.41"
    SV_PORT = 8080
    PORT = 6000
    if DEBUG:
        HOST = "127.0.0.1"
    else:
        HOST = "192.168.2.11"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", PORT))

    def send_motorCmd(motorCmd):
        motorCmd = STATE.brick_diff(motorCmd)
        if motorCmd == -1:
            data = "r"
        else:
            # ! Remove this
            # motorCmd = motorCmd[0]
            motorCmd = [i for j in motorCmd for i in j]
            data = json.dumps(motorCmd)
            print(data)
        if DEBUG:
            print(f"Send data: {data}")
        else:
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
        if USE_CAMERA:
            frame = get_frame()
            if frame is None:
                print("Failed to get frame")
                continue

            base64img = base64.b64encode(frame)
            encoded_data = json.dumps({"image": base64img.decode("utf-8")})
        if STATE.get_state() == NORMAL:
            if USE_CAMERA:
                response = post(
                    url + "detect",
                    data=encoded_data,
                    headers={"Content-Type": "application/json"},
                )
            else:
                response = post(url + "detect")
            if response.ok:
                motorCmd = response.json()
                send_motorCmd(motorCmd)
                time.sleep(delay_time / 1000)
            else:
                print("Failed to get response")
                time.sleep(delay_time / 5000)
        elif STATE.get_state() == PLAYING:
            if USE_CAMERA:
                response = post(
                    url + "image",
                    data=encoded_data,
                    headers={"Content-Type": "application/json"},
                )
            else:
                response = post(url + "image")

            response = post(url + "playing")
            motorCmd = response.json()
            send_motorCmd(motorCmd)
            time.sleep(delay_time / 1000)
        else:
            # print("Control state")
            if USE_CAMERA:
                response = post(
                    url + "image",
                    data=encoded_data,
                    headers={"Content-Type": "application/json"},
                )
            else:
                response = post(url + "image")
            time.sleep(delay_time / 1000)

    if USE_CAMERA:
        cap.release()
    if not DEBUG:
        s.close()
