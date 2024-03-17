from sklearn.utils import resample
from lightgbm import plot_importance
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))


def plot_top_categories_conversion_rate(df, category_col, target_col, lower, upper):
    conversion_counts = df.groupby([category_col, target_col]).size().unstack(fill_value=0)
    conversion_rates = conversion_counts.div(conversion_counts.sum(axis=1), axis=0)
    top_categories = conversion_rates[True].sort_values(ascending=False).index
    top_categories = top_categories[lower:upper]
    top_conversion_rates = conversion_rates.loc[top_categories]

    top_conversion_rates.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'From{lower} to {upper}/{category_col} Conversion Rate')
    plt.xlabel(category_col)
    plt.ylabel('Conversion Rate')
    plt.xticks(rotation=45)
    plt.legend(title='Is Converted', labels=['False', 'True'])
    plt.tight_layout()
    plt.show()


def calculate_roc_curve(gt, prediction, plot):
    # ROC Curve
    thresholds = np.linspace(0, 1, 100)
    TPRs, FPRs = [], []
    distances = []

    for threshold in thresholds:
        TP = np.sum(np.logical_and(gt == 1, prediction >= threshold))
        FN = np.sum(np.logical_and(gt == 1, prediction < threshold))
        FP = np.sum(np.logical_and(gt == 0, prediction >= threshold))
        TN = np.sum(np.logical_and(gt == 0, prediction < threshold))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPRs.append(TPR)
        FPRs.append(FPR)

        # (0,1)까지의 유클리드 거리 계산
        distance = np.sqrt((FPR - 0) ** 2 + (TPR - 1) ** 2)
        distances.append(distance)

    # 최소 거리와 해당하는 임계값 찾기
    min_distance_index = np.argmin(distances)
    optimal_threshold = thresholds[min_distance_index]

    if plot:
        # ROC curve 그리기
        plt.plot(FPRs, TPRs, label='ROC curve')
        plt.plot([0, 1], [0, 1], '--', label='Random')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    print(f'Optimal Threshold: {optimal_threshold}')

    return optimal_threshold


def plot_importace_lightGBM(bst):
    plt.figure(figsize=(10, 8))
    plot_importance(bst, importance_type='split')
    plt.title('Feature Importance by Weight')
    plt.show()

    plt.figure(figsize=(10, 8))
    plot_importance(bst, importance_type='gain')
    plt.title('Feature Importance by Gain')
    plt.show()


def sampling(train, target_name, size):
    X = train.drop(columns=[target_name])
    y = train[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    minority_indexes = np.where(y_train == 1)[0]

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

    borderline_smote = BorderlineSMOTE(random_state=42)
    X_line, y_line = borderline_smote.fit_resample(X_train, y_train)

    smote_data = pd.concat([X_smote, y_smote], axis=1)
    smote_data = smote_data[smote_data[target_name] == True & (~smote_data.index.isin(minority_indexes))]
    smote_data = resample(smote_data, replace=True, n_samples=1000, random_state=123)

    adasyn_data = pd.concat([X_adasyn, y_adasyn], axis=1)
    adasyn_data = adasyn_data[adasyn_data[target_name] == True & (~adasyn_data.index.isin(minority_indexes))]
    adasyn_data = resample(adasyn_data, replace=True, n_samples=1000, random_state=123)

    line_data = pd.concat([X_line, y_line], axis=1)
    line_data = line_data[line_data[target_name] == True & (~line_data.index.isin(minority_indexes))]
    line_data = resample(line_data, replace=True, n_samples=8000, random_state=123)

    synthetic_sample = pd.concat([smote_data, adasyn_data, line_data], axis=0)
    X_syn = synthetic_sample.drop(columns=[target_name])
    y_syn = synthetic_sample[target_name]

    X_res = pd.concat([X_train, X_syn], axis=0)
    y_res = pd.concat([y_train, y_syn], axis=0)

    return X_res, y_res, X_test, y_test
