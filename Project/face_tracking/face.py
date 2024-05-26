import cv2
import dlib
import os
from gaze_tracking.gaze_tracking import GazeTracking
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh


class Face:

    def __init__(self):
        self.frame = None
        self.landmarks = None
        self.mediapipe_landmarks = None
        self.face = None
        self.gaze_tracker = GazeTracking()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "../models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                    min_detection_confidence=0.5, refine_landmarks=True)


    def annotate(self):
        """
        Returns the main frame with face landmarks highlighted
        """
        # x, y, w, h = self.face.left(), self.face.top(), self.face.width(), self.face.height()
        # cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Landmark indices for eyes (36-47)
        frame = self.frame.copy()
        height, width = frame.shape[:2]
        for n in range(0, self.landmarks.num_parts):
            # if n in self.gaze_tracker.eye_left.LEFT_EYE_POINTS or n in self.gaze_tracker.eye_right.RIGHT_EYE_POINTS:
            #     continue
            x_eye, y_eye = self.landmarks.part(n).x, self.landmarks.part(n).y
            cv2.circle(frame, (x_eye, y_eye), 2, (0, 0, 255), -1)
        # if self.mediapipe_landmarks:
        #     for idx, n in enumerate(self.mediapipe_landmarks.landmark):
        #         if idx in self.gaze_tracker.eye_left.mediapipe_left_eye or idx in self.gaze_tracker.eye_right.mediapipe_right_eye:
        #             x = int(n.x * width)
        #             y = int(n.y * height)
        #             cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        #frame = self.gaze_tracker.annotated_frame(frame)
        return frame

    def analyze(self, frame, face):
        """
         Saves the current frame and analyzes the face for landmarks. Updates gaze tracker

         Arguments:
             frame The current frame
             face The detected face
        """
        self.frame = frame
        self.face = face

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #dlib
        landmarks = self._predictor(gray_frame, face)
        self.landmarks = landmarks

        # Use Mediapipe for landmark detection
        self.mediapipe_landmarks = self.get_landmarks_mediapipe(frame)

        self.gaze_tracker.analyze(landmarks, self.mediapipe_landmarks, frame)


    def get_landmarks_mediapipe(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None



