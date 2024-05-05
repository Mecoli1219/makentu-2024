import streamlit as st
import numpy as np
from PIL import Image
import io
from mqtt import get_mqtt_client
import time
import requests
from streamlit_vertical_slider import vertical_slider
import json
import base64
import cv2

VIEWER_WIDTH = 640
BRICK_NUM = 7

url = "http://10.20.2.41"
port = 8080


def byte_array_to_pil_image(byte_array):
    return Image.open(io.BytesIO(byte_array))


def get_random_numpy():
    """Return a dummy frame."""
    return np.random.randint(0, 100, size=(32, 32))


def onModeChange():
    state = 0
    if st.session_state.mode == "Playing":
        state = 1
    elif st.session_state.mode == "Control":
        state = 2
    requests.post(f"{url}:{port}/state", data=str(state))


def onFormSubmit():
    result = []
    for i in range(BRICK_NUM):
        u = st.session_state[f"u{i}"] or 0
        b = st.session_state[f"b{i}"] or 0
        result.append([u, b])
    requests.post(f"{url}:{port}/control", json=result)


st.title("Monitor")

if "frame" not in st.session_state:
    st.session_state.frame = get_random_numpy()
viewer = st.image(st.session_state.frame, width=VIEWER_WIDTH)
st.toggle("Debug", False, key="debug")

# select box
choice = st.radio(
    "Select Mode", ["Normal", "Playing", "Control"], on_change=onModeChange, key="mode"
)

if choice == "Control":
    st.header("Control Panel")
    with st.form(key="columns_in_form"):

        u_cols = st.columns(BRICK_NUM)
        for i, col in enumerate(u_cols):
            with col:
                vertical_slider(
                    key=f"u{i}",
                    height=100,  # Optional - Defaults to 300
                    max_value=3,  # Defaults to 10
                )

        b_cols = st.columns(BRICK_NUM)
        for i, col in enumerate(b_cols):
            with col:
                vertical_slider(
                    key=f"b{i}",
                    height=100,  # Optional - Defaults to 300
                    max_value=3,  # Defaults to 10
                    default_value=0,
                )
        submitted = st.form_submit_button("Submit", on_click=onFormSubmit)


def on_message(client, userdata, msg):
    if msg.topic == "image" and not st.session_state.debug:
        image = json.loads(msg.payload)["image"]
        image = base64.b64decode(image.encode("utf-8"))
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape((780, 680, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.session_state.frame = image
        viewer.image(st.session_state.frame, width=VIEWER_WIDTH)
    if msg.topic == "debug-image" and st.session_state.debug:
        image = json.loads(msg.payload)["image"]
        image = base64.b64decode(image.encode("utf-8"))
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape((780, 680, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.session_state.frame = image
        viewer.image(st.session_state.frame, width=VIEWER_WIDTH)


def main():

    client = get_mqtt_client()
    client.on_message = on_message
    client.connect("127.0.0.1", port=1883)
    client.subscribe("image", 0)
    client.subscribe("debug-image", 0)
    time.sleep(4)  # Wait for connection setup to complete
    client.loop_forever()


if __name__ == "__main__":
    main()
