import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


class DetectionModel:
    def __init__(self, model_path="pose_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        self.detector = detector

    def detect(self, image):
        return self.detector.detect(image)

    def result_image(self, image, detection_result):
        return draw_landmarks_on_image(image[:, :, :3], detection_result)


if __name__ == "__main__":
    imgBaseDir = "./testImg/"
    outputDir = "./output/"
    # imgList = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    imgList = ["img7.png", "img8.png"]
    model = DetectionModel()

    for imgPath in imgList:
        image = mp.Image.create_from_file(imgBaseDir + imgPath)
        detection_result = model.detect(image)
        annotated_image = model.result_image(image.numpy_view(), detection_result)
        pltImg = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(outputDir + imgPath, pltImg)
