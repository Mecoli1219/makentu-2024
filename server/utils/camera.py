import cv2
import mediapipe as mp

from pose import DetectionModel

# capture image from mac os camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
model = DetectionModel()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    media_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = model.detect(media_frame)
    result_image = model.result_image(frame, detection_result)
    cv2.imshow("frame", result_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
