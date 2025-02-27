import json
import os
import joblib
from pydantic import TypeAdapter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_auc_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from app.configuration.configuration_model import Configuration
from app.main import video_analysis
from imblearn.over_sampling import SMOTE
import numpy as np


RANDOM_STATE = 42


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

    X = np.array(X)  # Ensure X is a NumPy array
    y = np.array(y)  # Ensure y is a NumPy array

    # Split the data before applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    # Apply SMOTE only to the training set
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardization before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection using SelectKBest (tune k dynamically)
    k_best = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
    X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
    X_test_selected = k_best.transform(X_test_scaled)

    # Apply PCA to keep 95% variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    # Define hyperparameter grids
    param_grid_rf = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
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
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize models
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    svc = SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced')
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    # Use RandomizedSearchCV for hyperparameter tuning
    grid_search_rf = RandomizedSearchCV(rf, param_grid_rf, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=10)
    grid_search_svc = RandomizedSearchCV(svc, param_grid_svc, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=10)
    grid_search_gb = RandomizedSearchCV(gb, param_grid_gb, cv=5, n_jobs=-1, random_state=RANDOM_STATE, n_iter=10)

    # Fit models
    grid_search_rf.fit(X_train_pca, y_train)
    grid_search_svc.fit(X_train_pca, y_train)
    grid_search_gb.fit(X_train_pca, y_train)

    # Best models
    best_rf = grid_search_rf.best_estimator_
    best_svc = grid_search_svc.best_estimator_
    best_gb = grid_search_gb.best_estimator_

    print(f"Best RF: {grid_search_rf.best_params_}, Score: {grid_search_rf.best_score_}")
    print(f"Best SVC: {grid_search_svc.best_params_}, Score: {grid_search_svc.best_score_}")
    print(f"Best GB: {grid_search_gb.best_params_}, Score: {grid_search_gb.best_score_}")

    # Save models
    joblib.dump(best_rf, 'best_rf_model.pkl')
    joblib.dump(best_svc, 'best_svc_model.pkl')
    joblib.dump(best_gb, 'best_gb_model.pkl')

    # Evaluate models
    models = {'Random Forest': best_rf, 'SVC': best_svc, 'Gradient Boosting': best_gb}
    X_test_inputs = {'Random Forest': X_test_pca, 'SVC': X_test_pca, 'Gradient Boosting': X_test_pca}

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (name, model) in enumerate(models.items()):
        ConfusionMatrixDisplay.from_estimator(model, X_test_inputs[name], y_test, ax=axes[i], cmap='Blues')
        axes[i].title.set_text(f'{name} Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # Print classification reports
    for name, model in models.items():
        y_pred = model.predict(X_test_inputs[name])
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

    # Plot ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        y_probs = model.predict_proba(X_test_inputs[name])[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_probs, ax=ax, name=f"{name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")

    plt.title('ROC Curves')
    plt.show()


if __name__ == "__main__":
    main()