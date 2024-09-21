import cv2
import dlib
import os
from app.blink_detection.blink_tracking import BlinkTracking
from app.gaze_detection.gaze_tracking import GazeTracking

import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh


class Face:

    def __init__(self):
        self.frame = None
        self.landmarks = None
        self.mediapipe_landmarks = None
        self.face = None
        self.blink_tracker = BlinkTracking()
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

        frame = self.frame.copy()
        for n in range(0, self.landmarks.num_parts):

            x_eye, y_eye = self.landmarks.part(n).x, self.landmarks.part(n).y
            cv2.circle(frame, (x_eye, y_eye), 2, (0, 0, 255), -1)

        return frame

    def analyze(self, frame, face):
        """
         Saves the current frame and analyzes the face for landmarks. Updates trackers
        """
        self.frame = frame
        self.face = face

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #dlib
        landmarks = self._predictor(gray_frame, face)
        self.landmarks = landmarks

        # Use Mediapipe for landmark detection
        self.mediapipe_landmarks = self.get_landmarks_mediapipe(frame)

        self.blink_tracker.analyze(landmarks, self.mediapipe_landmarks, frame)
        self.gaze_tracker.analyze(landmarks, self.mediapipe_landmarks, frame)


    def get_landmarks_mediapipe(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None



