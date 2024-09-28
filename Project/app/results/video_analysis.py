from dataclasses import dataclass, field
import logging

from app.blink_detection.analyses.blink_analyses_results import TotalBlinkResults

@dataclass
class VideoAnalyses:
    blinking_analyses: TotalBlinkResults = field(default_factory=TotalBlinkResults)
    total_segments: int = 0
    total_frames: int = 0
    total_time: float = 0
    avg_segment_duration: float = 0

    def log(self):
        # Log overall analysis results
        logging.info("Overall Segment Analysis Results:")
        logging.info(f"Total Segments: {self.total_segments}")
        logging.info(f"Total Frames: {self.total_frames}")
        logging.info(f"Total Time: {self.total_time:.2f} seconds")
        logging.info(f"Average Segment Duration: {self.avg_segment_duration:.2f} seconds")

        # Log Dlib blink results
        logging.info("Dlib Blink Analysis Results:")
        logging.info(f"Total Blinks: {self.blinking_analyses.dlib_blink_count}")
        logging.info(f"Blink Rate: {self.blinking_analyses.dlib_blink_rate:.2f}")
        logging.info(f"Total Double Blinks: {self.blinking_analyses.dlib_double_blinks}")

        # Log Mediapipe blink results
        logging.info("Mediapipe Blink Analysis Results:")
        logging.info(f"Total Blinks: {self.blinking_analyses.mediapipe_blink_count}")
        logging.info(f"Blink Rate: {self.blinking_analyses.mediapipe_blink_rate:.2f}")
        logging.info(f"Total Double Blinks: {self.blinking_analyses.mediapipe_double_blinks}")

    def print_analysis(self):
        # Print overall analysis results
        print("Overall Segment Analysis Results:")
        print(f"Total Segments: {self.total_segments}")
        print(f"Total Frames: {self.total_frames}")
        print(f"Total Time: {self.total_time:.2f} seconds")
        print(f"Average Segment Duration: {self.avg_segment_duration:.2f} seconds")

        # Print Dlib blink results
        print("Dlib Blink Analysis Results:")
        print(f"Total Blinks: {self.blinking_analyses.dlib_blink_count}")
        print(f"Blink Rate: {self.blinking_analyses.dlib_blink_rate:.2f}")
        print(f"Average Dlib All Blink Rate: {self.blinking_analyses.dlib_all_blinks_rate:.2f}")
        print(f"Average Dlib All Blink Rate no doubles: {self.blinking_analyses.dlib_blinks_no_double_rate:.2f}")
        print(f"Total Double Blinks: {self.blinking_analyses.dlib_double_blinks}")
        for i, segment in enumerate(self.blinking_analyses.segment_result):
            print(f"Segment {i + 1}:")
            print(f"  Dlib Blink Count: {segment.dlib_blink_analysis.total_blink_count}")
            print(f"  Dlib Blink double blink Count: {segment.dlib_blink_analysis.total_double_blinks}")
            print(f"  All Dlib Blink Rate: {segment.dlib_blink_analysis.all_blinks_rate:.2f}")
            print(f"  All Dlib Blink Rate not double blinks: {segment.dlib_blink_analysis.blinks_no_double_rate:.2f}")


        # Print Mediapipe blink results
        print("Mediapipe Blink Analysis Results:")
        print(f"Total Blinks: {self.blinking_analyses.mediapipe_blink_count}")
        print(f"Blink Rate: {self.blinking_analyses.mediapipe_blink_rate:.2f}")
        print(f"Average Mediapipe All Blink Rate: {self.blinking_analyses.mediapipe_all_blinks_rate:.2f}")
        print(f"Average Mediapipe All Blink Rate no doubles: {self.blinking_analyses.mediapipe_blinks_no_double_rate:.2f}")
        print(f"Total Double Blinks: {self.blinking_analyses.mediapipe_double_blinks}")
        # Print Mediapipe blink results by segment
        for i, segment in enumerate(self.blinking_analyses.segment_result):
            print(f"Segment {i + 1}:")
            print(f"  Mediapipe Blink Count: {segment.mediapipe_blink_analysis.total_blink_count}")
            print(f"  Mediapipe Blink  double blink Count: {segment.mediapipe_blink_analysis.total_double_blinks}")
            print(f"  All Mediapipe Blink Rate: {segment.mediapipe_blink_analysis.all_blinks_rate:.2f}")
            print(f"  All Mediapipe Blink Rate not double blinks: {segment.mediapipe_blink_analysis.blinks_no_double_rate:.2f}")


     
     


        
        