from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, model_validator


class BlinkDetectionParameters(BaseModel):
    threshold: float
    max_double_blink_interval: float
    min_peak_value: float = 0.5
    max_second_dist: float = 0.4
    cutoff_scale: float = 0.8


class Configuration(BaseModel):
    show_progress_bar: bool
    show_video: bool
    save_to_file: bool
    results_directory: str
    image_directory: str
    video_file: str
    save_no_face_frames: bool
    enable_logging: bool = False
    log_file: Optional[str] = None
    blink_detection_parameters: BlinkDetectionParameters

    @model_validator(mode='after')
    def validate_logging(cls, values):
        enable_logging = values.enable_logging
        log_file = values.log_file

        if enable_logging and not log_file:
            raise ValueError("If enable_logging is True, log_file must be specified")
        if log_file and not enable_logging:
            values.enable_logging = True

        return values

    @classmethod
    def model_validate(cls, data):
        return super().model_validate(data)
