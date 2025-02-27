import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Correct import for StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

feature_labels = [
    "Blinks Rate", "Mean Blink Duration", "Avg BPM", "Avg SNR",
    "Median BPM", "Std BPM", "Min BPM", "Max BPM",
    "Median SNR", "Std SNR", "Min SNR", "Max SNR",
    "Unknown Gaze Rate", "Unknown Face Rate"
]

# Define classifier instances.
classifiers = {
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    'SVC': SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGB': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
}

# Pipeline variant: only SelectKBest.
def pipeline_kbest(model):
    return ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)),
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('classifier', model)
    ])

# Pipeline variant: SelectKBest followed by PCA.
def pipeline_kbest_pca(model):
    return ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)),
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('pca', PCA()),
        ('classifier', model)
    ])

# Pipeline variant: SelectKBest followed by RFE.
def pipeline_kbest_rfe(model):
    # For RFE, we use a RandomForest as the estimator for feature ranking.
    base_estimator = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    return ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)),
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('rfe', RFE(estimator=base_estimator)),
        ('classifier', model)
    ])

# Dictionary to hold pipeline variants.
pipeline_variants = {
    'KBest': pipeline_kbest,
    'KBest_PCA': pipeline_kbest_pca,
    'KBest_RFE': pipeline_kbest_rfe
}

# Preprocessing parameter grid for SelectKBest.
preproc_param_grid = {
    'kbest__k': [5, 10, 12],
}

# Classifier-specific parameter grids.
# Note: all keys for classifier parameters are prefixed with 'classifier__'
classifier_param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    },
    'SVC': {
        'classifier__C': [0.1, 1, 10, 100, 1000],
        'classifier__gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__learning_rate': [0.1, 0.05, 0.01, 0.005],
        'classifier__max_depth': [3, 4, 5, 6],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 5, 10]
    },
    'XGB': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
    }
}

# Additional parameter grids for PCA and RFE steps.
pca_param_grid = {
    'pca__n_components': [0.95, 0.99]
}
rfe_param_grid = {
    'rfe__n_features_to_select': [3, 8, 10]
}


def main():
    # Load data (expects a file 'data_results.joblib' with (X, y) tuple).
    if os.path.exists('data_results.joblib'):
        X, y = joblib.load('data_results.joblib')
    else:
        raise FileNotFoundError("The file 'data_results.joblib' does not exist.")
    
    print(f"Total samples: {len(X)}")
    df = pd.DataFrame(X, columns=feature_labels)
    y = np.array(y)
    
    # Combine and shuffle data.
    df['label'] = y
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    y = df['label'].values
    df = df.drop(columns=['label'])
    
    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    # Initialize variables to track best metrics.
    best_accuracy = 0
    best_recall_1 = 0
    best_precision_1 = 0
    best_f1_1 = 0
    best_acc_model = ""
    best_recall_model = ""
    best_precision_model = ""
    best_f1_model = ""
    best_overall_model = ""
    best_overall_score = 0  # Combined score as sum of metrics
    best_overall_model_accuracy = 0

    # Prepare subplots for confusion matrices: rows = pipeline variants, columns = classifiers.
    fig, axes = plt.subplots(nrows=len(pipeline_variants), ncols=len(classifiers), figsize=(20, 15))
    
    # Ensure axes is always a 2D array.
    if len(pipeline_variants) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(classifiers) == 1:
        axes = np.expand_dims(axes, axis=1)
    
    results = {}
    pipe_names = list(pipeline_variants.keys())
    clf_names = list(classifiers.keys())
    
    # Loop over each pipeline variant and classifier.
    for i, pipe_name in enumerate(pipe_names):
        for j, clf_name in enumerate(clf_names):
            print(f"\nTesting {clf_name} with pipeline {pipe_name}")
            ax = axes[i, j]
            
            # Build the pipeline.
            pipeline = pipeline_variants[pipe_name](classifiers[clf_name])
            
            # Start with common preprocessing parameters.
            param_grid = preproc_param_grid.copy()
            # Update with classifier parameters.
            param_grid.update(classifier_param_grids[clf_name])
            # If using PCA or RFE, add their parameters.
            if pipe_name == 'KBest_PCA':
                param_grid.update(pca_param_grid)
            if pipe_name == 'KBest_RFE':
                param_grid.update(rfe_param_grid)
            
            # Setup grid search.
            grid_search = RandomizedSearchCV(
                pipeline,
                param_grid,
                cv=5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                n_iter=20  # Adjust as needed.
            )
            
            # Fit grid search.
            grid_search.fit(X_train, y_train)
            
            # Predict on the test set.
            y_pred = grid_search.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[(clf_name, pipe_name)] = (grid_search.best_params_, report)
            
            print("Best parameters:")
            print(grid_search.best_params_)
            print("Classification report:")
            print(classification_report(y_test, y_pred))
            
            # Plot the confusion matrix.
            ConfusionMatrixDisplay.from_estimator(
                grid_search.best_estimator_, X_test, y_test, ax=ax, cmap='Blues'
            )
            ax.set_title(f"{clf_name} - {pipe_name}")
            
            # Extract metrics for class "1" (assuming class "1" is the positive class).
            accuracy = report["accuracy"]
            recall_1 = report.get("1", {}).get("recall", 0)
            precision_1 = report.get("1", {}).get("precision", 0)
            f1_1 = report.get("1", {}).get("f1-score", 0)
            
            # Update best metrics.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_acc_model = f"{clf_name} - {pipe_name}"
            if recall_1 > best_recall_1:
                best_recall_1 = recall_1
                best_recall_model = f"{clf_name} - {pipe_name}"
            if precision_1 > best_precision_1:
                best_precision_1 = precision_1
                best_precision_model = f"{clf_name} - {pipe_name}"
            if f1_1 > best_f1_1:
                best_f1_1 = f1_1
                best_f1_model = f"{clf_name} - {pipe_name}"
            
            # Compute an overall score (simple sum of metrics).
            overall_score = accuracy + recall_1 + precision_1 + f1_1
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_overall_model = f"{clf_name} - {pipe_name}"
                best_overall_model_accuracy = accuracy
    
    # Print best metric summary.
    print(f"\n🔥 Best Accuracy: {best_accuracy:.4f} ({best_acc_model})")
    print(f"🔥 Best Recall for Class 1: {best_recall_1:.4f} ({best_recall_model})")
    print(f"🔥 Best Precision for Class 1: {best_precision_1:.4f} ({best_precision_model})")
    print(f"🔥 Best F1-Score for Class 1: {best_f1_1:.4f} ({best_f1_model})")
    print(f"🔥 Overall Best Model (Combined Score): {best_overall_model} with score {best_overall_score:.4f} with accuracy of {best_overall_model_accuracy:.4f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
