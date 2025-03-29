import os
import json
import joblib
from pydantic import TypeAdapter
from app.configuration.configuration_model import Configuration
from app.main import video_analysis

def classify_video(video_path: str) -> bool:
    """
    Classifies a video as real (False) or fake (True).
    
    Args:
        video_path (str): Path to the video file to classify
        
    Returns:
        bool: True if the video is classified as fake, False if real
    """
    # Load the configuration
    config_path = os.path.join(os.path.dirname(__file__), 'configuration', 'params.json')
    with open(config_path, 'r') as file:
        data = json.load(file)
        config = TypeAdapter(Configuration).validate_python(data)
    
    # Process the video
    results = video_analysis(video_path, config)
    if results is None:
        raise ValueError("Could not process video - no face detected or other error")
    
    # Get the features in the correct format
    X = [results.get_results()]  # Wrap in list since we're classifying a single video
    print('Donde analizing video')
    # Load the model
    model_path = 'best_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file 'best_model.joblib' not found. Please train the model first.")
    
    model = joblib.load(model_path)

    print('Opened model')
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Print results
    result = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[1] if prediction == 1 else probability[0]
    print(f"\nVideo Classification Result:")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
    
    return prediction == 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Classify a video as real or fake')
    parser.add_argument('video_path', help='Path to the video file to classify')
    args = parser.parse_args()
    
    try:
        is_fake = classify_video(args.video_path)
        exit(0 if not is_fake else 1)  # Exit with 0 for real, 1 for fake
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(2)  # Exit with 2 for error

if __name__ == "__main__":
    main() 