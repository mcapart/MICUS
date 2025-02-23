import json
import os
import cv2
import joblib
from pydantic import TypeAdapter
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from app.configuration.configuration_model import Configuration
from app.main import video_analysis

def load_videos(video_dir):
    videos = []
    labels = []
    for subfolder in ['real', 'fake']:
        subfolder_path = os.path.join(video_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
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
    for idx, video in enumerate(videos):
        results = video_analysis(video, config)
        if results is None:
            labels.pop(idx)
        else:
            all_results.append(results)

    
    
    # Flatten the results for classification
    X = [r.get_results() for r in all_results]
    y = labels 

    joblib.dump((X, y), 'data_results.joblib')
    return X, y

def main():
    if os.path.exists('data_results.joblib'):
        X, y = joblib.load('data_results.joblib')
    else:
        X, y = analyze_videos()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Train the classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy Random forest: {accuracy}")


     # Scale the data for SVC
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVC classifier
    clf2 = SVC()
    clf2.fit(X_train_scaled, y_train)
    y_pred = clf2.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy SVC: {accuracy}")


    clf3 = GradientBoostingClassifier()
    clf3.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy GradientBoostingClassifier: {accuracy}")


    #joblib.dump(clf, 'classifier_model.joblib')

    
    # Predict and evaluate

if __name__ == "__main__":
    main()