import cv2
from requests import post
import json
from json import JSONEncoder
import numpy as np
import paho.mqtt.client as mqtt
import time
from copy import deepcopy

cap = cv2.VideoCapture(0)

NORMAL = 0
PLAYING = 1
CONTROL = 2

delay_time = 500


class State:
    def __init__(self):
        self._state = NORMAL
        self.brick_num = 9
        self.brick_status = [[0, 0] for _ in range(self.brick_num)]
        self.brick_multiply = 10

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


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (640, 480))
    return frame


if __name__ == "__main__":

    HOST = "127.0.0.1"
    PORT = 6000

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.bind((HOST, PORT))
    # indata, addr = s.recvfrom(1024)
    # msg = indata.decode()[:-1]
    # print("recvfrom " + str(addr) + ": " + msg)
    msg = "Start"

    def send_motorCmd(motorCmd):
        motorCmd = STATE.brick_diff(motorCmd)
        if motorCmd == -1:
            data = "r"
        else:
            data = json.dumps(motorCmd)

        # s.sendto(data.encode(), addr)
        print(data)

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

    if msg == "Start":
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(host="127.0.0.1", port=1883)
        client.subscribe("state", 0)
        client.subscribe("control", 0)
        client.loop_start()

        url = "http://127.0.0.1:5000/"

        # capture image from mac os camera
        while True:
            frame = get_frame()
            if frame is None:
                print("Failed to get frame")
                continue

            numpyData = {"image": frame}
            encoded_data = json.dumps(numpyData, cls=NumpyArrayEncoder)
            if STATE.get_state() == NORMAL:
                response = post(
                    url + "detect",
                    data=encoded_data,
                    headers={"Content-Type": "application/json"},
                )
                if response.ok:
                    motorCmd = response.json()
                    send_motorCmd(motorCmd)
                    # time.sleep(delay_time / 1000)
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

    # s.close()
