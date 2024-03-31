from sklearn.metrics import *

import numpy as np


def calculate_nd(gt, output):
    errors = np.abs(np.array(gt) - np.array(output))
    mean = np.mean(errors)
    std_deviation = np.std(errors)

    nd_list = (errors - mean) / std_deviation
    nd = np.mean(nd_list)
    return nd


def calculate_mae(gt, output):
    mae = mean_absolute_error(gt, output)
    return mae


def calculate_rmse(gt, output):
    rmse = np.sqrt(mean_squared_error(gt, output))
    return rmse


def calculate_rmse_v2(gt, output):
    output = output.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    rmse = np.sqrt(mean_squared_error(gt, output))
    return rmse


def calculate_adjusted_r2_score(gt, output, n, p):
    r2 = r2_score(gt, output)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adjusted_r2
