

import logging
from app.blink_detection.analyses.blink_anomalies import BlinkAnomaliesEnum


class VideoAnomalies:
    dlib_blinking_anomalies: list[BlinkAnomaliesEnum] = []
    mediapipe_blinking_anomalies: list[BlinkAnomaliesEnum] = []

    def log(self):
        # Log the detected anomalies
        if not self.dlib_blinking_anomalies and not self.mediapipe_blinking_anomalies:
            logging.info("No anomalies detected.")  
        else:
            logging.info("Dlib Blinking Anomalies: %s", self.dlib_blinking_anomalies)
            logging.info("Mediapipe Blinking Anomalies: %s", self.mediapipe_blinking_anomalies)

    def printAnomalies(self):
        # Print the anomalies in a formatted way
        if not self.dlib_blinking_anomalies and not self.mediapipe_blinking_anomalies:
            print("No anomalies detected.")  
        else:
            print("Dlib Blinking Anomalies:", self.dlib_blinking_anomalies)  
            print("Mediapipe Blinking Anomalies:", self.mediapipe_blinking_anomalies)  