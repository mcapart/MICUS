import cv2
import dlib
import os
from gaze_tracking.gaze_tracking import GazeTracking


class Face:

    def __init__(self):
        self.frame = None
        self.landmarks = None
        self.face = None
        self.gaze_tracker = GazeTracking()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "../models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def annotate(self):
        """
        Returns the main frame with face landmarks highlighted
        """
        x, y, w, h = self.face.left(), self.face.top(), self.face.width(), self.face.height()
        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Landmark indices for eyes (36-47)
        frame = self.frame.copy()
        for n in range(0, self.landmarks.num_parts):
            x_eye, y_eye = self.landmarks.part(n).x, self.landmarks.part(n).y
            cv2.circle(frame, (x_eye, y_eye), 2, (0, 0, 255), -1)
        frame = self.gaze_tracker.annotated_frame(frame)
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
        landmarks = self._predictor(gray_frame, face)
        self.landmarks = landmarks
        self.gaze_tracker.analyze(landmarks, frame)



