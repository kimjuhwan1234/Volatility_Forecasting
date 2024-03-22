from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import numpy as np


def calculate_nd(actual_values, predicted_values):
    nd = np.abs(np.array(predicted_values) - np.array(actual_values)) / np.array(actual_values)
    return nd


def calculate_rmse(actual_values, predicted_values):

    rmse = np.sqrt(mean_squared_error(np.array([actual_values]), np.array([predicted_values])))
    return rmse


def calculate_rmse_v2(gt, output):
    output = output.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    rmse = np.sqrt(mean_squared_error(gt, output))
    return rmse


def calculate_adjusted_r2_score(output, gt, n, p):
    output = output.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    r2 = r2_score(gt, output)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adjusted_r2


def calculate_rou90(actual_values, predicted_values):
    relative_errors = (np.abs(np.array(predicted_values) - np.array(actual_values)) / np.array(actual_values))
    rou90 = np.percentile(relative_errors, 90)
    return rou90


def calculate_rou50(actual_values, predicted_values):
    relative_errors = np.abs(np.array(predicted_values) - np.array(actual_values)) / np.array(actual_values)
    rou50 = np.percentile(relative_errors, 50)
    return rou50
