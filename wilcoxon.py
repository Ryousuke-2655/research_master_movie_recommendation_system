import numpy as np
from scipy.stats import wilcoxon

def maeAndRmse_wilcoxon(mae_list, rmse_list, MeanMae_list, MeanRmse_list):
    max_mean_mae_index = np.argmin(MeanMae_list)
    max_mae_list = mae_list[max_mean_mae_index]
    MeanMae_list[max_mean_mae_index] = 100

    second_max_mean_mae_index = np.argmin(MeanMae_list)
    second_max_mae_list = mae_list[second_max_mean_mae_index]

    max_mean_rmse_index = np.argmin(MeanRmse_list)
    max_rmse_list = rmse_list[max_mean_rmse_index]
    MeanRmse_list[max_mean_rmse_index] = 100

    second_max_mean_rmse_index = np.argmin(MeanRmse_list)
    second_max_rmse_list = rmse_list[second_max_mean_rmse_index]

    stat_mae, p_mae = wilcoxon(max_mae_list, second_max_mae_list)
    stat_rmse, p_rmse = wilcoxon(max_rmse_list, second_max_rmse_list)

    return stat_mae, stat_rmse, p_mae, p_rmse, max_mean_mae_index, second_max_mean_mae_index, max_mean_rmse_index, second_max_mean_rmse_index