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

    def get_gaze_intersection(self):
        if self.gaze is None:
            return None

        left_gaze_line, right_gaze_line = self.gaze.get_gaze_lines()

        if left_gaze_line is None or right_gaze_line is None:
            return None

        # Check if gaze lines intersect
        intersection = self.line_intersection(left_gaze_line, right_gaze_line)
        return intersection
        
    def detect_gaze_direction(self, intersection, frame_size):
        if intersection:
            # Determine gaze direction based on intersection point
            center_x = frame_size[1] / 2
            center_y = frame_size[0] / 2

            # Use a percentage of the frame dimensions as the threshold
            threshold_x = frame_size[1] * 0.1  # 10% of the frame width
            threshold_y = frame_size[0] * 0.1  # 10% of the frame height

            x_direction = None
            if intersection[0] < center_x - threshold_x:
                x_direction = GazeDirection.LEFT
            elif intersection[0] > center_x + threshold_x:
                x_direction = GazeDirection.RIGHT
            else :
                x_direction = GazeDirection.CENTER

            y_direction = None
            if intersection[1] < center_y - threshold_y:
                y_direction = GazeDirection.TOP
            elif intersection[1] > center_y + threshold_y:
                y_direction = GazeDirection.BOTTOM
            else:
                y_direction = GazeDirection.CENTER

            if x_direction == GazeDirection.CENTER and y_direction == GazeDirection.CENTER:
                return GazeDirection.CENTER
            elif x_direction == GazeDirection.LEFT and y_direction == GazeDirection.CENTER:
                return GazeDirection.LEFT
            elif x_direction == GazeDirection.RIGHT and y_direction == GazeDirection.CENTER:
                return GazeDirection.RIGHT
            elif x_direction == GazeDirection.CENTER and y_direction == GazeDirection.TOP:
                return GazeDirection.TOP
            elif x_direction == GazeDirection.CENTER and y_direction == GazeDirection.BOTTOM:
                return GazeDirection.BOTTOM
            elif x_direction == GazeDirection.LEFT and y_direction == GazeDirection.TOP:
                return GazeDirection.TOP_LEFT
            elif x_direction == GazeDirection.RIGHT and y_direction == GazeDirection.TOP:
                return GazeDirection.TOP_RIGHT
            elif x_direction == GazeDirection.LEFT and y_direction == GazeDirection.BOTTOM:
                return GazeDirection.BOTTOM_LEFT
            elif x_direction == GazeDirection.RIGHT and y_direction == GazeDirection.BOTTOM:
                return GazeDirection.BOTTOM_RIGHT
        else:
            return GazeDirection.UNKNOWN

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None  # Lines do not intersect

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

