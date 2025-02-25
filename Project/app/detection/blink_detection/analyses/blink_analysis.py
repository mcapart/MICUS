import numpy as np
from typing import  List, Tuple
from scipy.signal import find_peaks
from .blink_analyses_results import Blink, BlinkDuration, BlinkSegmentAnalysesResults, BlinkSegmentResult, TotalBlinkResults
from app.configuration.configuration_model import BlinkDetectionParameters
from app.results.video_tracking_result import  VideoTrackingResult,  FaceSegment
import matplotlib.pyplot as plt


class BlinkAnalyses:

    result: VideoTrackingResult
    blink_params:  BlinkDetectionParameters
    analysis_result: TotalBlinkResults
    fps: int
    max_frame_distance: int

    def __init__(self, result: VideoTrackingResult,  blink_params:  BlinkDetectionParameters, fps: int):
        self.result = result
        self.blink_params = blink_params
        self.fps = fps
        self.max_frame_distance = int(blink_params.max_second_dist * self.fps)

    #region "blink analysis"
    def analyze_video(self) -> TotalBlinkResults:
        segment_result = [self.analyze_segment(segment) for segment in self.result.segments]
        
        if not segment_result:
            return TotalBlinkResults()
        
        blink_count = sum(x.blink_analysis.total_blink_count for x in segment_result)
        blink_rate = np.mean([x.blink_analysis.blink_rate for x in segment_result])
        all_blinks_rate = np.mean([x.blink_analysis.all_blinks_rate for x in segment_result])
        blinks_no_double_rate = np.mean([x.blink_analysis.blinks_no_double_rate for x in segment_result if x.blink_analysis.blinks_no_double_rate> 0])
        double_blinks = sum(x.blink_analysis.total_double_blinks for x in segment_result)
        mean_druation = np.mean([x.blink_analysis.durations.mean_duration for x in segment_result if x.blink_analysis.durations is not None ])

        self.analysis_result = TotalBlinkResults(segment_result, 
                                                 blink_count, 
                                                 blink_rate, 
                                                 all_blinks_rate,
                                                 blinks_no_double_rate,
                                                 double_blinks,
                                                 mean_druation)
        return self.analysis_result
    
    def analyze_segment(self, segment: FaceSegment) -> BlinkSegmentResult:

        left_eye_ears = [frame.left_eye_ear for frame in segment.frames]
        right_eye_ears = [frame.right_eye_ear for frame in segment.frames]
        timestamps = [frame.timestamp_sec for frame in segment.frames]
        
        # Check if the segment has any frames
        if not timestamps:
            return BlinkSegmentResult()
        
        blink_results = self.detect_blinks(left_eye_ears, right_eye_ears, timestamps, threshold=self.blink_params.threshold, max_double_blink_interval=self.blink_params.max_double_blink_interval, min_peak_value=self.blink_params.min_peak_value,  cutoff_scale=self.blink_params.cutoff_scale)
     
        avg_ear = np.mean(left_eye_ears + right_eye_ears) if left_eye_ears and right_eye_ears else 0
        std_ear = np.std(left_eye_ears + right_eye_ears) if left_eye_ears and right_eye_ears else 0

        total_duration = timestamps[-1] - timestamps[0]
  
        return BlinkSegmentResult(avg_ear, std_ear, blink_results,
                                  duration=total_duration, 
                                  frame_count=len(segment.frames) )
   
    def detect_blinks(self, left_eye_ears: List[float], right_eye_ears: List[float], time_stamps: List[float], 
                    threshold: float = 0.3, max_double_blink_interval: float = 0.5,
                    min_peak_value: float = 0.5,  cutoff_scale: float = 0.8) -> BlinkSegmentAnalysesResults:
        if not left_eye_ears or not right_eye_ears or not time_stamps or len(time_stamps) < 2:
            return BlinkSegmentAnalysesResults()
        
        avg_derivative = self.calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
        all_peaks = self.calculate_derivatives_peaks(avg_derivative, threshold)

        midpoints = (np.array(time_stamps[:-1]) + np.array(time_stamps[1:])) / 2
        peak_times = midpoints[all_peaks]
        pairs = self.calculate_peak_pairs(all_peaks, midpoints, avg_derivative, min_peak_value, cutoff_scale)

        blinks = []
        double_blinks = 0
        last_blink_end = None
        all_blink_start_time = []
        blink_start_time_no_double = []

        for blink_start, blink_end in pairs:
            start_index = np.where(peak_times == blink_start)[0][0]
            end_index = np.where(peak_times == blink_end)[0][0]
            
            blinks.append(Blink(all_peaks[start_index], all_peaks[end_index], blink_start, blink_end))
            all_blink_start_time.append( blink_start)

            if last_blink_end is not None:
                time_between_blinks = blink_start - last_blink_end
                if time_between_blinks <= max_double_blink_interval:
                    double_blinks += 1
                else:
                    blink_start_time_no_double.append(blink_start)
            else:
                blink_start_time_no_double.append(blink_start)
            last_blink_end = blink_end

        total_duration = time_stamps[-1] - time_stamps[0]
        blink_count = len(blinks)
        blink_rate = (blink_count / total_duration) * 60  # blinks per minute
        time_between_all_blinks = [all_blink_start_time[i] - all_blink_start_time[i - 1] for i in range(1, len(all_blink_start_time))]
        time_between_real_blinks = [blink_start_time_no_double[i] - blink_start_time_no_double[i - 1] for i in range(1, len(blink_start_time_no_double))]
           # Calculate average time between blinks
        avg_time_between_blinks = np.mean(time_between_all_blinks) if time_between_all_blinks else 0
        avg_time_between_real_blinks = np.mean(time_between_real_blinks) if time_between_real_blinks else 0

        # Calculate blinks per minute
        blinks_per_minute = 60 / avg_time_between_blinks if avg_time_between_blinks > 0 else 0
        real_blinks_per_minute = 60 / avg_time_between_real_blinks if avg_time_between_real_blinks > 0 else 0


        durations = self.analyze_blink_durations(blinks)


        return BlinkSegmentAnalysesResults(total_blink_count=blink_count,
                                           blink_rate=blink_rate,
                                           total_double_blinks=double_blinks,
                                           blinks=blinks, 
                                           durations=durations,
                                           all_blinks_rate=blinks_per_minute,
                                           blinks_no_double_rate=real_blinks_per_minute
                                           )
        
     

    def calculate_avg_derivatives(self,left_eye_ears: List[float], right_eye_ears: List[float], time_stamps: List[float]) -> np.ndarray:
        if not time_stamps or len(time_stamps) < 2:
            return np.array([])  # Return an empty array if there are not enough timestamps
        
        time_stamps = np.array(time_stamps)
        avg_ears = self.calculate_avg_ears(left_eye_ears, right_eye_ears)
        return np.gradient(avg_ears, time_stamps)
    
    def calculate_avg_ears(self, left_eye_ears: List[float], right_eye_ears: List[float]):
        return (np.array(left_eye_ears) + np.array(right_eye_ears)) / 2.0

    def calculate_derivatives_peaks(self, avg_derivative: np.ndarray, threshold: float) -> np.ndarray:
        if len(avg_derivative) == 0:
            return np.array([])
        max_derivative = np.max(avg_derivative)
        pos_peaks, _ = find_peaks(avg_derivative, height=threshold * max_derivative, distance=10)
        neg_peaks, _ = find_peaks(-avg_derivative, height=threshold * (-np.min(avg_derivative)), distance=10)
        return np.sort(np.concatenate((pos_peaks, neg_peaks)))

    def calculate_peak_pairs(self, all_peaks: np.ndarray, midpoints: np.ndarray, avg_derivative: np.ndarray,
                            min_peak_value: float = 0.5,
                            cutoff_scale: float = 0.8) -> List[Tuple[float, float]]:
        blink_pairs = []
        max_derivative = np.max(avg_derivative)
        min_derivative = np.min(avg_derivative)
        distance = np.abs(max_derivative - min_derivative)

        for i in range(len(all_peaks) - 1):
            peak1, peak2 = all_peaks[i], all_peaks[i + 1]
            value_1, value_2 = avg_derivative[peak1], avg_derivative[peak2]
            height_diff = np.abs(value_1 - value_2)
            frame_distance = np.abs(peak1 - peak2)
            if frame_distance > self.max_frame_distance:
                continue

            scaled_cutoff = np.interp(frame_distance, [1, self.max_frame_distance], [1, distance * cutoff_scale])
            cutoff_distance = max(scaled_cutoff, 0.9)


            if ((value_1 < 0 and value_2 > 0) or (value_2 < 0 < value_1)) and \
            abs(value_1) > min_peak_value and abs(value_2) > min_peak_value and \
            height_diff >= cutoff_distance :
                blink_pairs.append((midpoints[peak1], midpoints[peak2]))

        return blink_pairs

 
    def analyze_blink_durations(self, blinks: List[Blink]) -> BlinkDuration:
        if not blinks or len(blinks) == 0:
            return BlinkDuration()

        durations = [blink.duration for blink in blinks]  

        return BlinkDuration(mean_duration=np.mean(durations), 
                            median_duration=np.median(durations),  
                            min_duration=np.min(durations),        
                            max_duration=np.max(durations),        
                            std_duration=np.std(durations))        
    


    #endregion "blink analysis"

 

   