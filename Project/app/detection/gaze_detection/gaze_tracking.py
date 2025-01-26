import os
import dlib

from .models.gaze_models import GazeDirection
from .gaze import Gaze


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    """

    def __init__(self):
        self.landmarks = None
        self.faces = []
        self.no_landmark = 0


        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "../../models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)
    
    def analyze(self,  landmarks, frame):
        self.landmarks = landmarks
        self.frame = frame

        if self.landmarks and self.frame is not None:
            self.gaze = Gaze(self.frame, self.landmarks)
        else:
            self.gaze = None

    def get_gaze_direction(self):
        if self.gaze is None:
            return GazeDirection.UNKNOWN

        left_gaze_line, right_gaze_line = self.gaze.get_gaze_lines()

        if left_gaze_line is None or right_gaze_line is None:
            return GazeDirection.UNKNOWN

        # Check if gaze lines intersect
        intersection = self.line_intersection(left_gaze_line, right_gaze_line)

        if intersection:
            # Determine gaze direction based on intersection point
            center_x = self.frame.shape[1] / 2
            if intersection[0] < center_x - 50:
                return GazeDirection.LEFT
            elif intersection[0] > center_x + 50:
                return GazeDirection.RIGHT
            else:
                return GazeDirection.CENTER
        else:
            # If lines don't intersect, use the average direction
            left_direction = GazeDirection.LEFT if left_gaze_line[1][0] - left_gaze_line[0][0] < 0 else GazeDirection.RIGHT
            right_direction = GazeDirection.LEFT if right_gaze_line[1][0] - right_gaze_line[0][0] < 0 else GazeDirection.RIGHT
            
            if left_direction == right_direction:
                return left_direction
            else:
                return GazeDirection.CENTER

    def line_intersection(self, line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        else:
            return None  # Lines don't intersect within the segments

