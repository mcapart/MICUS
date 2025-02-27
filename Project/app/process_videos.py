import json
import os

import joblib
from pydantic import TypeAdapter

from app.configuration.configuration_model import Configuration
from app.main import video_analysis


def load_videos(video_dir):
    videos = []
    labels = []
    for subfolder in ['fake', 'real']:
        subfolder_path = os.path.join(video_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            if not filename.endswith('.mp4'):
                continue
            label = 1 if subfolder == 'fake' else 0
            video_path = os.path.join(subfolder_path, filename)
            videos.append(video_path)
            labels.append(label)
    return videos, labels
    
def analyze_videos():
    video_dir = "/Users/micacapart/Documents/ITBA/dataset"
    videos, labels = load_videos(video_dir)

    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'params.json')

    with open(config_path, 'r') as file:
        data = json.load(file)
        config = TypeAdapter(Configuration).validate_python(data)
    
    all_results = []
    all_labels = []
    for idx, video in enumerate(videos):
        results = video_analysis(video, config)
        if results is not None:
            all_results.append(results)
            all_labels.append(labels[idx])

    
    
    # Flatten the results for classification
    X = [r.get_results() for r in all_results]
    y = all_labels 

    joblib.dump((X, y), 'data_results.joblib')
    return X, y

def main():
    if not os.path.exists('data_results.joblib'):
        analyze_videos()
    else:
        print('videos already analyzed')
