from dataclasses import dataclass, field
import logging
from typing import List

from app.detection import GazeSegmentAnalysesResult, TotalBlinkResults

    # all_blinks_rate
    # mean_duration: float = 0
    # avg bpm
    # avg snr
    # rate of unknown gaze unknown gazes / cant frames


@dataclass
class VideoAnalyses:
    blinking_analyses: TotalBlinkResults = field(default_factory=TotalBlinkResults)
    total_segments: int = 0
    total_frames: int = 0
    total_time: float = 0
    avg_segment_duration: float = 0
    segment_gaze_analyses: List[GazeSegmentAnalysesResult] =  field(default_factory=list)
    overall_gaze_analysis: GazeSegmentAnalysesResult =  field(default_factory=list)

    def log(self):
        # Log overall analysis results
        logging.info("Overall Segment Analysis Results:")
        logging.info(f"Total Segments: {self.total_segments}")
        logging.info(f"Total Frames: {self.total_frames}")
        logging.info(f"Total Time: {self.total_time:.2f} seconds")
        logging.info(f"Average Segment Duration: {self.avg_segment_duration:.2f} seconds")


        # Log Mediapipe blink results
        logging.info(" Blink Analysis Results:")
        logging.info(f"Total Blinks: {self.blinking_analyses.blink_count}")
        logging.info(f"Blink Rate: {self.blinking_analyses.blink_rate:.2f}")
        logging.info(f"Total Double Blinks: {self.blinking_analyses.double_blinks}")

        # Log gaze results
         # Log overall gaze analysis results
        logging.info("Overall Gaze Analysis Results:")
        logging.info(f"Direction Counts: {self.overall_gaze_analysis.direction_counts}")
        logging.info(f"Direction Percentages: {self.overall_gaze_analysis.direction_percentages}")
        logging.info(f"Most Common Direction: {self.overall_gaze_analysis.most_common_direction}")
        logging.info(f"Transitions: {self.overall_gaze_analysis.transitions}")
        logging.info(f"Durations: {self.overall_gaze_analysis.durations}")
        logging.info(f"Rapid Gaze Shifts: {self.overall_gaze_analysis.rapid_gaze_shifts}")

        # Log segment-wise gaze analysis results
        for i, segment_analysis in enumerate(self.segment_gaze_analyses):
            logging.info(f"Segment {i + 1} Gaze Analysis Results:")
            logging.info(f"Direction Counts: {segment_analysis.direction_counts}")
            logging.info(f"Direction Percentages: {segment_analysis.direction_percentages}")
            logging.info(f"Most Common Direction: {segment_analysis.most_common_direction}")
            logging.info(f"Transitions: {segment_analysis.transitions}")
            logging.info(f"Durations: {segment_analysis.durations}")
            logging.info(f"Rapid Gaze Shifts: {segment_analysis.rapid_gaze_shifts}")

    def print_analysis(self):
        # Print overall analysis results
        print("Overall Segment Analysis Results:")
        print(f"Total Segments: {self.total_segments}")
        print(f"Total Frames: {self.total_frames}")
        print(f"Total Time: {self.total_time:.2f} seconds")
        print(f"Average Segment Duration: {self.avg_segment_duration:.2f} seconds")


        # Print Mediapipe blink results
        print(" Blink Analysis Results:")
        print(f"Total Blinks: {self.blinking_analyses.blink_count}")
        print(f"Blink Rate: {self.blinking_analyses.blink_rate:.2f}")
        print(f"Average  All Blink Rate: {self.blinking_analyses.all_blinks_rate:.2f}")
        print(f"Average  All Blink Rate no doubles: {self.blinking_analyses.blinks_no_double_rate:.2f}")
        print(f"Total Double Blinks: {self.blinking_analyses.double_blinks}")
        # Print Mediapipe blink results by segment
        for i, segment in enumerate(self.blinking_analyses.segment_result):
            print(f"Segment {i + 1}:")
            print(f"   Blink Count: {segment.blink_analysis.total_blink_count}")
            print(f"   Blink  double blink Count: {segment.blink_analysis.total_double_blinks}")
            print(f"  All  Blink Rate: {segment.blink_analysis.all_blinks_rate:.2f}")
            print(f"  All  Blink Rate not double blinks: {segment.blink_analysis.blinks_no_double_rate:.2f}")

@dataclass
class VideoAnalysesResults:
    blinks_rate: float = 0
    mean_blink_duration: float = 0
    avg_bpm: float = 0
    avg_snr: float = 0
    median_bpm: float = 0
    std_bpm: float = 0
    min_bpm: float = 0
    max_bpm: float = 0

    median_snr: float = 0
    std_snr: float = 0
    min_snr: float = 0
    max_snr: float = 0
    unknown_gaze_rate: float = 0
    unknown_face_rate: float = 0

    def get_results(self):
        return [self.blinks_rate, self.mean_blink_duration, self.avg_bpm, self.avg_snr, self.unknown_gaze_rate, self.unknown_face_rate]
     


        
        