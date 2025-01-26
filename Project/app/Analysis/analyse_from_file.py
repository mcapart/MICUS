

import argparse
import json
import os

from pydantic import TypeAdapter

from app.analysis.vid_analysis import analyze_video
from app.analysis.utils import load_data_from_results
from app.configuration.configuration_model import Configuration

from app.results.video_tracking_result import VideoTrackingResult


def analyse_video_from_result_file(result:VideoTrackingResult, config: Configuration ):
    print('Starting analysis')
    analysis_results = analyze_video(result, config.blink_detection_parameters)
    analysis_results.print_analysis()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_path')
    args = parser.parse_args()
    file_path = args.file_path
    result = load_data_from_results(file_path) 
    config_path = os.path.join(os.path.dirname(__file__), '../configuration', 'params.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
        conf = TypeAdapter(Configuration).validate_python(data)
    
    analyse_video_from_result_file(result, config=conf)
