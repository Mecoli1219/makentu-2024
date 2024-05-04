import cv2
import mediapipe as mp
import numpy as np

from .pose import DetectionModel
from .labels import LABEL2NAME, NAME2LABEL
from .constant import NORMAL, PLAYING, CONTROL

MIDDLE = 0
LEFT = 1
RIGHT = -1


def landmark2np(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


class DecisionMaker:
    def __init__(self, num_brick=9, model_path="pose_landmarker.task"):
        self.detector = DetectionModel(model_path)
        self.name2label = NAME2LABEL
        self.label2name = LABEL2NAME
        self.num_brick = num_brick
        self.head_threshold = 1
        self.isSafe = True
        self.facePos = MIDDLE
        self.safeCount = 0
        self.safeThreshold = 5
        self.startSafety = False
        self.safetyUp = True
        self.detection_result = None
        self.brick_status = [[0, 0] for _ in range(num_brick)]
        self.max_level = 3
        self.prev_isSafe = True

        self._state = NORMAL
        self.playing_count = 0

    def headPos(self, pose_landmarks):
        # Get the position of the head
        nose = pose_landmarks[self.name2label["NOSE"]]
        left_eye = (
            pose_landmarks[self.name2label["LEFT_EYE"]]
            + pose_landmarks[self.name2label["LEFT_EYE_INNER"]]
            + pose_landmarks[self.name2label["LEFT_EYE_OUTER"]]
        ) / 3
        right_eye = (
            pose_landmarks[self.name2label["RIGHT_EYE"]]
            + pose_landmarks[self.name2label["RIGHT_EYE_INNER"]]
            + pose_landmarks[self.name2label["RIGHT_EYE_OUTER"]]
        ) / 3
        left_mouth = pose_landmarks[self.name2label["MOUTH_LEFT"]]
        right_mouth = pose_landmarks[self.name2label["MOUTH_RIGHT"]]
        head_pos = (nose + left_eye + right_eye + left_mouth + right_mouth) / 5
        return head_pos

    def bodyPos(self, pose_landmarks):
        # Get the position of the body
        left_shoulder = pose_landmarks[self.name2label["LEFT_SHOULDER"]]
        right_shoulder = pose_landmarks[self.name2label["RIGHT_SHOULDER"]]
        left_hip = pose_landmarks[self.name2label["LEFT_HIP"]]
        right_hip = pose_landmarks[self.name2label["RIGHT_HIP"]]
        body_pos = (left_shoulder + right_shoulder) / 2
        hip_pos = (left_hip + right_hip) / 2
        return left_shoulder, right_shoulder, body_pos, hip_pos

    def babyStatus(self, pose_landmarks, aspect_ratio):
        head_pos = self.headPos(pose_landmarks)
        l_shoulder, r_shoulder, body_pos, hip_pos = self.bodyPos(pose_landmarks)
        # Check whether face down
        if np.cross(l_shoulder - r_shoulder, body_pos - hip_pos)[2] > 0:
            isSafe = False
        else:
            isSafe = True

        # Check whether head is in the middle
        mid_lane = body_pos - hip_pos
        mid_lane[2] = 0
        mid_lane = mid_lane * aspect_ratio
        mid_lane = mid_lane / np.linalg.norm(mid_lane)
        head_lane = head_pos - body_pos
        head_lane[2] = 0
        head_lane = head_lane * aspect_ratio
        head_lane = head_lane / np.linalg.norm(head_lane)
        shoulder_lane = l_shoulder - r_shoulder
        shoulder_lane[2] = 0
        shoulder_lane = shoulder_lane * aspect_ratio
        shoulder_len = np.linalg.norm(shoulder_lane)
        shoulder_lane = shoulder_lane / shoulder_len

        displacement = (
            np.dot(
                np.cross(np.cross(mid_lane, head_lane), mid_lane),
                shoulder_lane,
            )
            / shoulder_len
        )
        if np.abs(displacement) < self.head_threshold * np.linalg.norm(aspect_ratio):
            facePos = MIDDLE
        elif displacement > 0:
            facePos = LEFT
        else:
            facePos = RIGHT

        return isSafe, facePos

    def voteResult(self, image):
        if image.shape[2] == 3:
            image = np.concatenate([image, np.ones_like(image[:, :, :1]) * 255], axis=2)
        images = [
            np.rot90(image, 1).copy(),
            np.rot90(image, 2).copy(),
            np.rot90(image, 3).copy(),
            image,
        ]
        results = [[], []]
        for img in images:

            media_frame = mp.Image(
                image_format=mp.ImageFormat.SRGBA, data=np.array(img)
            )
            detection_result = self.detector.detect(media_frame)
            pose_landmarks_list = detection_result.pose_landmarks
            if len(pose_landmarks_list) != 1:
                continue
            self.detection_result = detection_result
            pose_landmarks = pose_landmarks_list[0]
            pose_landmarks = [landmark2np(landmark) for landmark in pose_landmarks]

            if img.shape[0] > img.shape[1]:
                aspect_ratio = np.asarray([img.shape[1] / img.shape[0], 1, 1])
            else:
                aspect_ratio = np.asarray([1, img.shape[0] / img.shape[1], 1])
            isSafe, facePos = self.babyStatus(pose_landmarks, aspect_ratio)
            results[0].append(isSafe)
            results[1].append(facePos)
        if len(results[0]) == 0:
            return None, None

        isSafe = np.mean(results[0]) > 0.5
        facePos = max(set(results[1]), key=results[1].count)
        return isSafe, facePos

    def startSafetyWave(self, up_brick, down_brick):
        if up_brick < down_brick:
            if not self.startSafety:
                self.safetyUp = True
                self.startSafety = True
                for i in range(up_brick + 1, down_brick):
                    self.brick_status[i] = [0, 0]
            else:
                if self.safetyUp:
                    for i in range(down_brick - 1, up_brick, -1):
                        self.brick_status[i] = self.brick_status[i - 1]
                        if (
                            self.brick_status[down_brick - 1][0] == self.max_level
                            and self.brick_status[down_brick - 1][1] == self.max_level
                        ):
                            self.safetyUp = False
                else:
                    for i in range(up_brick + 1, down_brick):
                        self.brick_status[i] = self.brick_status[i + 1]
                        if (
                            self.brick_status[up_brick + 1][0] == 0
                            and self.brick_status[up_brick + 1][1] == 0
                        ):
                            self.safetyUp = True
        else:
            if not self.startSafety:
                self.safetyUp = True
                self.startSafety = True
                for i in range(down_brick + 1, up_brick):
                    self.brick_status[i] = [0, 0]
            else:
                if self.safetyUp:
                    for i in range(down_brick + 1, up_brick):
                        self.brick_status[i] = self.brick_status[i + 1]
                        if (
                            self.brick_status[down_brick + 1][0] == self.max_level
                            and self.brick_status[down_brick + 1][1] == self.max_level
                        ):
                            self.safetyUp = False
                else:
                    for i in range(up_brick - 1, down_brick, -1):
                        self.brick_status[i] = self.brick_status[i - 1]
                        if (
                            self.brick_status[up_brick - 1][0] == 0
                            and self.brick_status[up_brick - 1][1] == 0
                        ):
                            self.safetyUp = True
        return

    def plotBrickStatus(self, image):
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        border = 5
        # print(self.brick_status)
        for i in range(self.num_brick):
            for j in range(2):
                color = [0, 0, 0]
                if self.brick_status[i][j] in [1, 2, 3]:
                    color[self.brick_status[i][j] - 1] = 255
                    cv2.rectangle(
                        image,
                        (
                            int(i * image.shape[1] / self.num_brick) + border,
                            (image.shape[0] * j) // 2 + border,
                        ),
                        (
                            int((i + 1) * image.shape[1] / self.num_brick) - border,
                            (image.shape[0] * (j + 1)) // 2 - border,
                        ),
                        color,
                        border * 2,
                    )
                else:
                    cv2.rectangle(
                        image,
                        (
                            int(i * image.shape[1] / self.num_brick) + border,
                            (image.shape[0] * j) // 2 + border,
                        ),
                        (
                            int((i + 1) * image.shape[1] / self.num_brick) - border,
                            (image.shape[0] * (j + 1)) // 2 - border,
                        ),
                        (255, 255, 255),
                        border * 2,
                    )
        return image

    def makeDecision(self, image=None):
        if self._state != NORMAL:
            return None
        if image is not None:
            isSafe, facePos = self.voteResult(image)
            if isSafe is None:
                return None
            self.prev_isSafe = isSafe
            if isSafe:
                if self.safeCount > self.safeThreshold:
                    self.isSafe = True
                else:
                    self.safeCount += 1
                self.startSafety = False
            else:
                self.safeCount = 0
                self.isSafe = False
            self.facePos = facePos

        # TODO: Deal with orientation
        pose_landmarks = [
            landmark2np(landmark)
            for landmark in self.detection_result.pose_landmarks[0]
        ]
        left_shoulder, right_shoulder, body_pos, hip_pos = self.bodyPos(pose_landmarks)
        body_lane = body_pos - hip_pos

        if image.shape[0] > image.shape[1]:
            aspect_ratio = np.asarray([image.shape[1] / image.shape[0], 1, 1])
        else:
            aspect_ratio = np.asarray([1, image.shape[0] / image.shape[1], 1])
        body_x = body_lane[0] * aspect_ratio[0]
        body_y = body_lane[1] * aspect_ratio[1]
        if abs(body_x) > abs(body_y):
            pivot_brick = int(body_pos[0] * self.num_brick)
            pivot_brick = min(max(pivot_brick, 0), self.num_brick - 1)
            if body_x > 0:
                if body_y < 0:
                    for i in range(pivot_brick, self.num_brick):
                        self.brick_status[i] = [self.max_level - 1, self.max_level]
                    for i in range(pivot_brick):
                        self.brick_status[i] = [1, 0]
                else:
                    for i in range(pivot_brick, self.num_brick):
                        self.brick_status[i] = [self.max_level, self.max_level - 1]
                    for i in range(pivot_brick):
                        self.brick_status[i] = [0, 1]
            else:
                if body_y < 0:
                    for i in range(pivot_brick + 1):
                        self.brick_status[i] = [self.max_level - 1, self.max_level]
                    for i in range(pivot_brick + 1, self.num_brick):
                        self.brick_status[i] = [1, 0]
                else:
                    for i in range(pivot_brick + 1):
                        self.brick_status[i] = [self.max_level, self.max_level - 1]
                    for i in range(pivot_brick + 1, self.num_brick):
                        self.brick_status[i] = [0, 1]

        elif not self.isSafe:  # * Deal with unsafe situation
            if self.facePos >= MIDDLE:
                up_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["LEFT_SHOULDER"]
                ]
                down_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["RIGHT_SHOULDER"]
                ]
            elif self.facePos < MIDDLE:
                up_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["RIGHT_SHOULDER"]
                ]
                down_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["LEFT_SHOULDER"]
                ]

            up_brick = int(up_pos.x * self.num_brick)
            down_brick = int(down_pos.x * self.num_brick)

            # avoid the case that both shoulders are in the same brick
            if up_pos.x - down_pos.x > 0:
                down_brick = min(down_brick, up_brick - 1)

                for i in range(up_brick, self.num_brick):
                    self.brick_status[i] = [self.max_level, self.max_level]
                for i in range(down_brick + 1):
                    self.brick_status[i] = [0, 0]
                self.startSafetyWave(up_brick, down_brick)

            else:
                down_brick = max(down_brick, up_brick + 1)

                for i in range(down_brick, self.num_brick):
                    self.brick_status[i] = [0, 0]
                for i in range(up_brick + 1):
                    self.brick_status[i] = [self.max_level, self.max_level]
                self.startSafetyWave(up_brick, down_brick)
        elif self.facePos != MIDDLE:
            # * Deal with safe situation
            base_level = 1
            self.brick_status = [
                [base_level, base_level] for _ in range(self.num_brick)
            ]
            if self.facePos == LEFT:
                up_shoulder_pos = left_shoulder
                up_shoulder_brick_x = int(up_shoulder_pos[0] * self.num_brick)
                up_shoulder_brick_y = int(up_shoulder_pos[1] > 0.5)
                up_eye_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["LEFT_EYE"]
                ]
                up_eye_brick_x = int(up_eye_pos.x * self.num_brick)
                up_eye_brick_y = int(up_eye_pos.y > 0.5)

                down_shoulder_pos = right_shoulder
                down_shoulder_brick_x = int(down_shoulder_pos[0] * self.num_brick)
                down_shoulder_brick_y = int(down_shoulder_pos[1] > 0.5)
                down_eye_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["RIGHT_EYE"]
                ]
                down_eye_brick_x = int(down_eye_pos.x * self.num_brick)
                down_eye_brick_y = int(down_eye_pos.y > 0.5)

            elif self.facePos == RIGHT:
                up_shoulder_pos = right_shoulder
                up_shoulder_brick_x = int(up_shoulder_pos[0] * self.num_brick)
                up_shoulder_brick_y = int(up_shoulder_pos[1] > 0.5)
                up_eye_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["RIGHT_EYE"]
                ]
                up_eye_brick_x = int(up_eye_pos.x * self.num_brick)
                up_eye_brick_y = int(up_eye_pos.y > 0.5)

                down_pos = left_shoulder
                down_shoulder_brick_x = int(down_pos[0] * self.num_brick)
                down_shoulder_brick_y = int(down_pos[1] > 0.5)
                down_eye_pos = self.detection_result.pose_landmarks[0][
                    self.name2label["LEFT_EYE"]
                ]
                down_eye_brick_x = int(down_eye_pos.x * self.num_brick)
                down_eye_brick_y = int(down_eye_pos.y > 0.5)

            # (body_pos[0], body_pos[1]), (hip_pos[0], hip_pos[1]), (which_x_1, 25), (which_x_2, 75) in the same line
            which_x_1 = (0.25 - body_pos[1]) / (hip_pos[1] - body_pos[1]) * (
                hip_pos[0] - body_pos[0]
            ) + body_pos[0]
            which_x_2 = (0.75 - body_pos[1]) / (hip_pos[1] - body_pos[1]) * (
                hip_pos[0] - body_pos[0]
            ) + body_pos[0]
            which_brick_1 = int(which_x_1 * self.num_brick)
            which_brick_2 = int(which_x_2 * self.num_brick)

            if up_shoulder_pos[0] - down_shoulder_pos[0] > 0:
                # Right should pull up
                brick_list = [
                    [which_brick_1 + 1, self.num_brick],
                    [which_brick_2 + 1, self.num_brick],
                ]
                if up_shoulder_brick_y == 0:
                    brick_list[0].append(up_shoulder_brick_x)
                else:
                    brick_list[1].append(up_shoulder_brick_x)
                if up_eye_brick_y == 0:
                    brick_list[0].append(up_eye_brick_x)
                else:
                    brick_list[1].append(up_eye_brick_x)
                start_1 = min(brick_list[0])
                start_2 = min(brick_list[1])
                for i in range(start_1):
                    self.brick_status[i][0] = base_level
                for i in range(start_1, self.num_brick):
                    self.brick_status[i][0] = base_level + 1
                for i in range(start_2):
                    self.brick_status[i][1] = base_level
                for i in range(start_2, self.num_brick):
                    self.brick_status[i][1] = base_level + 1

            else:
                # Left should pull up
                brick_list = [[which_brick_1 - 1, -1], [which_brick_2 - 1, -1]]
                if up_shoulder_brick_y == 0:
                    brick_list[0].append(up_shoulder_brick_x)
                else:
                    brick_list[1].append(up_shoulder_brick_x)
                if up_eye_brick_y == 0:
                    brick_list[0].append(up_eye_brick_x)
                else:
                    brick_list[1].append(up_eye_brick_x)
                start_1 = min(max(brick_list[0]), self.num_brick - 1)
                start_2 = min(max(brick_list[1]), self.num_brick - 1)

                for i in range(start_1 + 1, self.num_brick):
                    self.brick_status[i][0] = base_level
                for i in range(start_1 + 1):
                    self.brick_status[i][0] = base_level + 1
                for i in range(start_2 + 1, self.num_brick):
                    self.brick_status[i][1] = base_level
                for i in range(start_2 + 1):
                    self.brick_status[i][1] = base_level + 1

        else:
            base_level = 0
            self.brick_status = [
                [base_level, base_level] for _ in range(self.num_brick)
            ]
        return self.brick_status

    def getState(self):
        return self._state

    def _getPlayingPattern(self):
        # [[1,1], [2,2], [3,3], [2,2],[1,1], ...]
        brick_state = []
        for i in range(self.num_brick):
            level = i % 4 + 1
            if level == 4:
                level = 2
            brick_state.append([level, level])
        return brick_state

    def updateState(self, state):
        if state in [NORMAL, PLAYING, CONTROL]:
            self._state = state
            if state == NORMAL:
                self.isSafe = True
                self.facePos = MIDDLE
                self.safeCount = 0
                self.startSafety = False
                self.safetyUp = True
                self.brick_status = [[0, 0] for _ in range(self.num_brick)]
                self.prev_isSafe = True
            if state == PLAYING:
                self.brick_status = self._getPlayingPattern()
                self.playing_count = 0
            if state == CONTROL:
                self.brick_status = [[0, 0] for _ in range(self.num_brick)]
        return self.brick_status

    def updatePlaying(self):
        if self._state == PLAYING:
            self.playing_count += 1
            self.playing_count %= 4
            level = self.playing_count + 1
            if level == 4:
                level = 2
            for i in range(self.num_brick - 1, 0, -1):
                self.brick_status[i][0] = self.brick_status[i - 1][0]
                self.brick_status[i][1] = self.brick_status[i - 1][1]
            self.brick_status[0] = [level, level]
        return self.brick_status

    def updateControl(self, control):
        if self._state == CONTROL:
            self.brick_status = control
        return self.brick_status


if __name__ == "__main__":
    imgBaseDir = "./testImg/"
    outputDir = "./output/"
    imgList = []
    # imgList.extend(["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"])
    # imgList.extend(["img1-1.jpg", "img1-2.jpg", "img1-3.jpg"])
    # imgList.extend(["img2-1.jpg", "img2-2.jpg", "img2-3.jpg"])
    # imgList.extend(["img3-1.jpg", "img3-2.jpg", "img3-3.jpg"])
    # imgList.extend(["img4-1.jpg", "img4-2.jpg", "img4-3.jpg"])
    # imgList.extend(["45img1.png", "45img2.png", "45img3.png", "45img4.png"])
    imgList.extend(["45img1-1.png", "45img1-2.png", "45img1-3.png"])
    imgList.extend(["45img2-1.png", "45img2-2.png", "45img2-3.png"])
    imgList.extend(["45img3-1.png", "45img3-2.png", "45img3-3.png"])
    imgList.extend(["45img4-1.png", "45img4-2.png", "45img4-3.png"])
    decisionMaker = DecisionMaker()

    for imgPath in imgList:
        print(f"Current Image: {imgPath}")
        # decisionMaker.isSafe = True
        image = mp.Image.create_from_file(imgBaseDir + imgPath)
        while True:
            decisionMaker.makeDecision(image.numpy_view())

            img = decisionMaker.plotBrickStatus(image.numpy_view())
            cv2.imshow("frame", img)
            # wait 1 second
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                break

        detection_result = decisionMaker.detection_result
        if detection_result is None:
            continue

        annotated_image = decisionMaker.detector.result_image(
            image.numpy_view(), detection_result
        )
        pltImg = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(outputDir + imgPath, pltImg)

    # pose_landmarks_list = detection_result.pose_landmarks
    # print(len(pose_landmarks_list))

    # image = plot_landmarks(image.numpy_view(), pose_landmarks_list[0])
    # cv2.imshow("frame", image)
    # cv2.waitKey(0)
