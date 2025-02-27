import json
import os
import joblib
from pydantic import TypeAdapter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.model_selection import  RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from app.configuration.configuration_model import Configuration
from app.main import video_analysis
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


RANDOM_STATE = 42

feature_labels = [
    "Blinks Rate", "Mean Blink Duration", "Avg BPM", "Avg SNR",
    "Median BPM", "Std BPM", "Min BPM", "Max BPM",
    "Median SNR", "Std SNR", "Min SNR", "Max SNR",
    "Unknown Gaze Rate", "Unknown Face Rate"
]

# Define hyperparameter grids
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],  # NEW
    'class_weight': ['balanced', 'balanced_subsample']
}

param_grid_svc = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree': [2, 3, 4]
}

param_grid_gb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'max_depth': [3, 4, 5, 6],
    'max_features': ['sqrt', 'log2', None],  # NEW
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Expanded
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10]  # Expanded
}

param_grid_xbg = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [3, 5, 7],  # Maximum depth of each tree
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
    'subsample': [0.7, 0.8, 1.0],  # Fraction of training samples
    'colsample_bytree': [0.7, 0.8, 1.0],  # Fraction of features per tree
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required for a split
}



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
    # Load data if available, otherwise analyze videos
    if os.path.exists('data_results.joblib'):
        X, y = joblib.load('data_results.joblib')
    else:
        X, y = analyze_videos()
    
    print(f"Total samples: {len(X)}")
    df = pd.DataFrame(X, columns=feature_labels)
    y = np.array(y)  # Ensure y is a NumPy array

    # Reduce VIF before scaling
    
    # Split the data before applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    # Apply SMOTE only to the training set
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardization before PCA & RFE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection using SelectKBest (tune k dynamically)
    k_best = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
    X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
    X_test_selected = k_best.transform(X_test_scaled)


    # Apply PCA (keep 95% variance)
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    # Apply RFE (Recursive Feature Elimination)
    n_features_to_select = min(8, X_train.shape[1])
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=n_features_to_select)
    X_train_rfe = rfe.fit_transform(X_train_selected, y_train)
    X_test_rfe = rfe.transform(X_test_selected)
    selected_features_rfe = np.array(feature_labels)[np.where(rfe.support_)[0]]
    print("\nFeatures selected by RFE:", selected_features_rfe)

    # Define models
    models = {
        "Base": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                 SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
                 GradientBoostingClassifier(random_state=RANDOM_STATE),
                 XGBClassifier(random_state=RANDOM_STATE,  eval_metric="logloss")
                 ],
        "PCA": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                 SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
                 GradientBoostingClassifier(random_state=RANDOM_STATE),
                 XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
                 ],
        "RFE": [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                 SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
                 GradientBoostingClassifier(random_state=RANDOM_STATE),
                 XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
                 ]
    }

    x_train_by = {
        "Base": X_train_selected,
        "PCA": X_train_pca,
        "RFE": X_train_rfe
    }

    best_models = {
        "Base": None,
        "PCA": None,
        "RFE": None
    }

    for name, model in models.items():
        print(f"\n{name} Grid search")
        grid_search_rf = RandomizedSearchCV(model[0], param_grid_rf, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=50)
        grid_search_svc = RandomizedSearchCV(model[1], param_grid_svc, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=50)
        grid_search_gb = RandomizedSearchCV(model[2], param_grid_gb, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=50)
        grid_search_xgb = RandomizedSearchCV(model[3], param_grid_xbg, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=50)
        x_train_name = x_train_by[name]
        # Fit models
        grid_search_rf.fit(x_train_name, y_train)
        grid_search_svc.fit(x_train_name, y_train)
        grid_search_gb.fit(x_train_name, y_train)
        grid_search_xgb.fit(x_train_name, y_train)

        # Best models
        best_rf = grid_search_rf.best_estimator_
        best_svc = grid_search_svc.best_estimator_
        best_gb = grid_search_gb.best_estimator_
        best_xgb = grid_search_xgb.best_estimator_

        best_models[name] = [best_rf, best_svc, best_gb, best_xgb]
    

    # Evaluate models
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))  # 3x3 grid for confusion matrices
    X_test_inputs = {"Base": X_test_selected, "PCA": X_test_pca, "RFE": X_test_rfe}

    # Initialize variables to track the best metrics
    best_accuracy = 0
    best_recall_1 = 0
    best_precision_1 = 0
    best_f1_1 = 0
    best_acc_model = ""
    best_recall_model = ""
    best_precision_model = ""
    best_f1_model = ""
    best_overall_model = ""
    best_overall_score = 0  # This will track the score for the overall best model

    for i, (name, models_list) in enumerate(best_models.items()):
        X_test_data = X_test_inputs[name]

        for j, model in enumerate(models_list):
            y_pred = model.predict(X_test_data)

            # Compute classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Extract metrics for class 1
            accuracy = report["accuracy"]
            recall_1 = report["1"]["recall"]  # Recall for class 1
            precision_1 = report["1"]["precision"]  # Precision for class 1
            f1_1 = report["1"]["f1-score"]  # F1-score for class 1

            # Track best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_acc_model = f"{name} - {type(model).__name__}"

            # Track best recall for class 1
            if recall_1 > best_recall_1:
                best_recall_1 = recall_1
                best_recall_model = f"{name} - {type(model).__name__}"

            # Track best precision for class 1
            if precision_1 > best_precision_1:
                best_precision_1 = precision_1
                best_precision_model = f"{name} - {type(model).__name__}"

            # Track best f1-score for class 1
            if f1_1 > best_f1_1:
                best_f1_1 = f1_1
                best_f1_model = f"{name} - {type(model).__name__}"

            # Calculate an overall score (you can customize the weights of each metric if needed)
            overall_score = accuracy + recall_1 + precision_1 + f1_1  # Simple sum of all metrics

            # Track the overall best model
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_overall_model = f"{name} - {type(model).__name__}" 


            # Plot Confusion Matrix
            ConfusionMatrixDisplay.from_estimator(model, X_test_data, y_test, ax=axes[i, j], cmap='Blues')
            axes[i, j].title.set_text(f'{name} - {type(model).__name__}')

            # Print classification report
            print(f"\nClassification Report for {name} - {type(model).__name__}:")
            print(classification_report(y_test, y_pred))



    # Print overall best metrics
    print(f"\n🔥 Best Accuracy: {best_accuracy:.4f} ({best_acc_model})")
    print(f"🔥 Best Recall for Class 1: {best_recall_1:.4f} ({best_recall_model})")
    print(f"🔥 Best Precision for Class 1: {best_precision_1:.4f} ({best_precision_model})")
    print(f"🔥 Best F1-Score for Class 1: {best_f1_1:.4f} ({best_f1_model})")

    # Print the overall best model based on the combined score
    print(f"\n🔥 Overall Best Model (Combined Score): {best_overall_model} with score {best_overall_score:.4f}")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()