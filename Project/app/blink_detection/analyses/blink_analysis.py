import numpy as np
from typing import  Dict, List, Tuple
from scipy.signal import find_peaks


from app.blink_detection.analyses.blink_analyses_results import Blink, BlinkDuration, BlinkSegmentAnalysesResults, BlinkSegmentResult, TotalBlinkResults
from app.blink_detection.analyses.blink_anomalies import BlinkAnomaliesEnum
from app.configuration.configuration_model import BlinkDetectionParameters, LandmarkModel
from app.results.video_tracking_result import FaceSegment, VideoTrackingResult


class BlinkAnalyses:

    result: VideoTrackingResult
    blink_params: Dict[LandmarkModel, BlinkDetectionParameters]
    analysis_result: TotalBlinkResults

    def __init__(self, result: VideoTrackingResult,  blink_params: Dict[LandmarkModel, BlinkDetectionParameters]):
        self.result = result
        self.blink_params = blink_params

    #region "blink analysis"
    def analyze_video(self) -> TotalBlinkResults:
        segment_result = [self.analyze_segment(segment) for segment in self.result.segments]
        
        if not segment_result:
            return TotalBlinkResults()
        dlib_blink_count = sum(x.dlib_blink_analysis.total_blink_count for x in segment_result)
        dlib_blink_rate= np.mean([x.dlib_blink_analysis.blink_rate for x in segment_result])
        dlib_all_blinks_rate= np.mean([x.dlib_blink_analysis.all_blinks_rate for x in segment_result])
        dlib_blinks_no_double_rate= np.mean([x.dlib_blink_analysis.blinks_no_double_rate for x in segment_result if x.dlib_blink_analysis.blinks_no_double_rate > 0])
        dlib_double_blinks =  sum(x.dlib_blink_analysis.total_double_blinks for x in segment_result)
        dlib_mean_druation = np.mean([x.dlib_blink_analysis.durations.mean_duration for x in segment_result ])
        
        mediapipe_blink_count = sum(x.mediapipe_blink_analysis.total_blink_count for x in segment_result)
        mediapipe_blink_rate = np.mean([x.mediapipe_blink_analysis.blink_rate for x in segment_result])
        mediapipe_all_blinks_rate = np.mean([x.mediapipe_blink_analysis.all_blinks_rate for x in segment_result])
        mediapipe_blinks_no_double_rate = np.mean([x.mediapipe_blink_analysis.blinks_no_double_rate for x in segment_result if x.mediapipe_blink_analysis.blinks_no_double_rate> 0])
        mediapipe_double_blinks = sum(x.mediapipe_blink_analysis.total_double_blinks for x in segment_result)
        mediapie_mean_druation = np.mean([x.mediapipe_blink_analysis.durations.mean_duration for x in segment_result ])

        self.analysis_result = TotalBlinkResults(segment_result, 
                                                 dlib_blink_count, 
                                                 dlib_blink_rate, 
                                                 dlib_all_blinks_rate,
                                                 dlib_blinks_no_double_rate,
                                                 dlib_double_blinks,
                                                 dlib_mean_druation, 
                                                 mediapipe_blink_count, 
                                                 mediapipe_blink_rate, 
                                                 mediapipe_all_blinks_rate,
                                                 mediapipe_blinks_no_double_rate,
                                                 mediapipe_double_blinks,
                                                 mediapie_mean_druation)
        return self.analysis_result
    
    def analyze_segment(self, segment: FaceSegment) -> BlinkSegmentResult:
        dlib_left_eye_ears = [frame.left_eye_ear for frame in segment.frames]
        dlib_right_eye_ears = [frame.right_eye_ear for frame in segment.frames]

        left_eye_mediapipe_ears = [frame.left_eye_mediapipe_ear for frame in segment.frames]
        right_eye_mediapipe_ears = [frame.right_eye_mediapipe_ear for frame in segment.frames]
        timestamps = [frame.timestamp_sec for frame in segment.frames]
        
        # Check if the segment has any frames
        if not timestamps:
            return BlinkSegmentResult()
        
        params = self.blink_params[LandmarkModel.DLIB]
        dlib_blink_results = self.detect_blinks(dlib_left_eye_ears, dlib_right_eye_ears, timestamps, 
                                    threshold=params.threshold, 
                                    max_double_blink_interval=params.max_double_blink_interval, 
                                    min_peak_value=params.min_peak_value, 
                                    max_frame_distance=params.max_frame_distance, 
                                    cutoff_scale=params.cutoff_scale)
    

        # Mediapipe blink analysis
        params = self.blink_params[LandmarkModel.MEDIAPIPE]
        mediapipe_blink_results = self.detect_blinks(left_eye_mediapipe_ears, right_eye_mediapipe_ears, timestamps, threshold=params.threshold, max_double_blink_interval=params.max_double_blink_interval, min_peak_value=params.min_peak_value, max_frame_distance=params.max_frame_distance, cutoff_scale=params.cutoff_scale)
     

        avg_dlib_ear =  np.mean(dlib_left_eye_ears + dlib_right_eye_ears) if dlib_left_eye_ears and dlib_right_eye_ears else 0
        std_dlib_ear = np.std(dlib_left_eye_ears + dlib_right_eye_ears) if dlib_left_eye_ears and dlib_right_eye_ears else 0
        avg_mediapipe_ear = np.mean(left_eye_mediapipe_ears + right_eye_mediapipe_ears) if left_eye_mediapipe_ears and right_eye_mediapipe_ears else 0
        std_mediapipe_ear = np.std(left_eye_mediapipe_ears + right_eye_mediapipe_ears) if left_eye_mediapipe_ears and right_eye_mediapipe_ears else 0

        total_duration = timestamps[-1] - timestamps[0]
  
        return BlinkSegmentResult(avg_dlib_ear, 
                                  std_dlib_ear, 
                                  avg_mediapipe_ear, 
                                  std_mediapipe_ear, 
                                  dlib_blink_analysis=dlib_blink_results, 
                                  mediapipe_blink_analysis=mediapipe_blink_results, 
                                  duration=total_duration, 
                                  frame_count=len(segment.frames) )
   
    def detect_blinks(self, left_eye_ears: List[float], right_eye_ears: List[float], time_stamps: List[float], 
                    threshold: float = 0.3, max_double_blink_interval: float = 0.5,
                    min_peak_value: float = 0.5, max_frame_distance: int = 10, cutoff_scale: float = 0.8) -> BlinkSegmentAnalysesResults:
        if not left_eye_ears or not right_eye_ears or not time_stamps or len(time_stamps) < 2:
            return BlinkSegmentAnalysesResults()
        
        avg_derivative = self.calculate_avg_derivatives(left_eye_ears, right_eye_ears, time_stamps)
        all_peaks = self.calculate_derivatives_peaks(avg_derivative, threshold)
        midpoints = (np.array(time_stamps[:-1]) + np.array(time_stamps[1:])) / 2
        peak_times = midpoints[all_peaks]
        pairs = self.calculate_peak_pairs(all_peaks, midpoints, avg_derivative, min_peak_value, max_frame_distance, cutoff_scale)

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
                            max_frame_distance: int = 10,
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

            scaled_cutoff = np.interp(frame_distance, [1, max_frame_distance], [1, distance * cutoff_scale])
            cutoff_distance = max(scaled_cutoff, 1)

            if ((value_1 < 0 and value_2 > 0) or (value_2 < 0 < value_1)) and \
            abs(value_1) > min_peak_value and abs(value_2) > min_peak_value and \
            height_diff >= cutoff_distance and frame_distance <= max_frame_distance:
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

    #region "anomalies"

    def detect_anomaly(self):
        if self.analysis_result is None:
            return
        
        #The rate a human blinks in a minute is between 5 and 30 times. More normally 15 to 20
        min_rate = 5
        max_rate = 30

        min_blink_duration = 0.1  # 100 ms
        max_blink_duration = 0.4  # 400 ms
        tolerance = 0.02           # 20 milliseconds tolerance

        double_blink_threshold = 0.1  # 10% of total blinks


        dlib_anomalies: list[BlinkAnomaliesEnum] = []
        #region dlib
        if self.analysis_result.dlib_blink_count == 0:
            dlib_anomalies.append(BlinkAnomaliesEnum.NO_BLINK)
        no_double_rate = self.analysis_result.dlib_blinks_no_double_rate
        if no_double_rate < min_rate or no_double_rate > max_rate:
            dlib_anomalies.append(BlinkAnomaliesEnum.BLINKING_RATE_WRONG)

       

        if self.analysis_result.dlib_mean_duration < (min_blink_duration - tolerance):
            dlib_anomalies.append(BlinkAnomaliesEnum.BLINKS_TOO_SHORT)
        elif self.analysis_result.dlib_mean_duration > (max_blink_duration + tolerance):
            dlib_anomalies.append(BlinkAnomaliesEnum.BLINKS_TOO_LONG)

    
        if self.analysis_result.dlib_blink_count > 0:
            double_blink_ratio = self.analysis_result.dlib_double_blinks / self.analysis_result.dlib_blink_count
            if double_blink_ratio > double_blink_threshold:
                dlib_anomalies.append(BlinkAnomaliesEnum.HIGH_DOUBLE_BLINK_FREQUENCY)
        elif self.analysis_result.dlib_double_blinks > 0:
            dlib_anomalies.append(BlinkAnomaliesEnum.HIGH_DOUBLE_BLINK_FREQUENCY)

        blink_rates = [segment.dlib_blink_analysis.blinks_no_double_rate for segment in self.analysis_result.segment_result]
        blink_rate_std = np.std(blink_rates)
        blink_rate_mean = np.mean(blink_rates)
    
        # Check if the standard deviation is more than 50% of the mean
        if blink_rate_std > 0.5 * blink_rate_mean:
            dlib_anomalies.append(BlinkAnomaliesEnum.INCONSISTENT_BLINK_RATE)
        
        # New check for discrepancy between both eyes
        for segment in self.result.segments:
            dlib_left_eye_ears = [frame.left_eye_ear for frame in segment.frames]
            dlib_right_eye_ears = [frame.right_eye_ear for frame in segment.frames]
            left_ear_mean = np.mean(dlib_left_eye_ears)
            right_ear_mean = np.mean(dlib_right_eye_ears)
            
            # Check if the difference between left and right eye EAR is more than 20%
            if abs(left_ear_mean - right_ear_mean) / ((left_ear_mean + right_ear_mean) / 2) > 0.2:
                dlib_anomalies.append(BlinkAnomaliesEnum.EYE_DISCREPANCY)
                break

        mediapipe_anomalies: list[BlinkAnomaliesEnum] = []
        #endregion 
        #region mediapipe
        if self.analysis_result.mediapipe_blink_count == 0:
            mediapipe_anomalies.append(BlinkAnomaliesEnum.NO_BLINK)
        no_double_rate = self.analysis_result.mediapipe_blinks_no_double_rate
        if no_double_rate < min_rate or no_double_rate > max_rate:
            mediapipe_anomalies.append(BlinkAnomaliesEnum.BLINKING_RATE_WRONG)

        if self.analysis_result.mediapipe_mean_duration < (min_blink_duration - tolerance):
            mediapipe_anomalies.append(BlinkAnomaliesEnum.BLINKS_TOO_SHORT)
        elif self.analysis_result.mediapipe_mean_duration > (max_blink_duration + tolerance):
            mediapipe_anomalies.append(BlinkAnomaliesEnum.BLINKS_TOO_LONG)

    
        if self.analysis_result.mediapipe_blink_count > 0:
            double_blink_ratio = self.analysis_result.mediapipe_double_blinks / self.analysis_result.mediapipe_blink_count
            if double_blink_ratio > double_blink_threshold:
                dlib_anomalies.append(BlinkAnomaliesEnum.HIGH_DOUBLE_BLINK_FREQUENCY)
        elif self.analysis_result.mediapipe_double_blinks > 0:
            dlib_anomalies.append(BlinkAnomaliesEnum.HIGH_DOUBLE_BLINK_FREQUENCY)

        blink_rates = [segment.mediapipe_blink_analysis.blinks_no_double_rate for segment in self.analysis_result.segment_result]
        blink_rate_std = np.std(blink_rates)
        blink_rate_mean = np.mean(blink_rates)
    
        # Check if the standard deviation is more than 50% of the mean
        if blink_rate_std > 0.5 * blink_rate_mean:
            mediapipe_anomalies.append(BlinkAnomaliesEnum.INCONSISTENT_BLINK_RATE)
        
        # New check for discrepancy between both eyes
        for segment in self.result.segments:
            mediapipe_left_eye_ears = [frame.left_eye_mediapipe_ear for frame in segment.frames]
            mediapipe_right_eye_ears = [frame.right_eye_mediapipe_ear for frame in segment.frames]
            left_ear_mean = np.mean(mediapipe_left_eye_ears)
            right_ear_mean = np.mean(mediapipe_right_eye_ears)
            
            # Check if the difference between left and right eye EAR is more than 20%
            if abs(left_ear_mean - right_ear_mean) / ((left_ear_mean + right_ear_mean) / 2) > 0.2:
                mediapipe_anomalies.append(BlinkAnomaliesEnum.EYE_DISCREPANCY)
                break
        #endregion
        
        return dlib_anomalies, mediapipe_anomalies

        

    #enregion "anomailes"

   