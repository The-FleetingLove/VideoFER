import torch
import numpy as np


def pcc_ccc_func(labels_th, scores_th):
    std_l_v = torch.std(labels_th[:, 0])
    std_p_v = torch.std(scores_th[:, 0])
    std_l_a = torch.std(labels_th[:, 1])
    std_p_a = torch.std(scores_th[:, 1])
    mean_l_v = torch.mean(labels_th[:, 0])
    mean_p_v = torch.mean(scores_th[:, 0])
    mean_l_a = torch.mean(labels_th[:, 1])
    mean_p_a = torch.mean(scores_th[:, 1])

    PCC_v = torch.mean((labels_th[:, 0] - mean_l_v) * (scores_th[:, 0] - mean_p_v)) / (std_l_v * std_p_v)
    PCC_a = torch.mean((labels_th[:, 1] - mean_l_a) * (scores_th[:, 1] - mean_p_a)) / (std_l_a * std_p_a)
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / (std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v - mean_p_v).pow(2))
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / (std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a - mean_p_a).pow(2))

    PCC_loss = 1.0 - (PCC_v + PCC_a) / 2
    CCC_loss = 1.0 - (CCC_v + CCC_a) / 2
    return PCC_loss, CCC_loss, PCC_v, PCC_a, CCC_v, CCC_a


def RMSE_func(mse_value):
    return torch.sqrt(mse_value)


def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)
    return ccc


def VA_metric(x, y):
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    return items
