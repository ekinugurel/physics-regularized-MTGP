"""
Author: Ekin Ugurel

Citation: 
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
import numpy as np
import similaritymeasures as sm
import pandas as pd
import gpytorch
from scipy.stats import norm

def absolute_percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_pred)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(absolute_percentage_error(y_true, y_pred))

def max_absolute_percentage_error(y_true, y_pred):
    return np.max(absolute_percentage_error(y_true, y_pred))

def total_absolute_percentage_error(y_true, y_pred):
    return np.sum(absolute_percentage_error(y_true, y_pred))

def evaluate(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAD': median_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'MAXAPE': max_absolute_percentage_error(y_true, y_pred),
        'TAPE': total_absolute_percentage_error(y_true, y_pred)
    }

def average_eval(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    eval1 = evaluate(y_true_lat, y_pred_lat)
    eval2 = evaluate(y_true_lon, y_pred_lon)
    
    averaged = list()
    for i, j in zip(eval1.values(), eval2.values()):
        averaged.append(np.sqrt(i**2 + j**2))
        
    return {
        'MAE': averaged[0],
        'RMSE': averaged[1],
        'MAD': averaged[2],
        'MAPE': averaged[3],
        'MAXAPE': averaged[4],
        'TAPE': averaged[5]
    }

def evaluate_similarity(lat_tc, pred_mean, y_test_scaled):
    """
    Evaluate the similarity between the predicted and true curves
    using various metrics.
    
    """
    preds_lat = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), pred_mean[:,0].reshape(-1,1)))
    test_lat = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), y_test_scaled[:,0].reshape(-1,1)))

    preds_lon = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), pred_mean[:,1].reshape(-1,1)))
    test_lon = np.hstack((pd.Series(lat_tc.index).values.reshape(-1,1), y_test_scaled[:,1].reshape(-1,1)))

    # quantify the difference between the two curves using PCM
    pcm_lat = sm.pcm(preds_lat, test_lat)
    pcm_lon = sm.pcm(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Discrete Frechet distance
    df_lat = sm.frechet_dist(preds_lat, test_lat)
    df_lon = sm.frechet_dist(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # area between two curves
    area_lat = sm.area_between_two_curves(preds_lat, test_lat)
    area_lon = sm.area_between_two_curves(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Curve Length based similarity measure
    cl_lat = sm.curve_length_measure(preds_lat, test_lat)
    cl_lon = sm.curve_length_measure(preds_lon, test_lon)

    # quantify the difference between the two curves using
    # Dynamic Time Warping distance
    dtw_lat, d_lat = sm.dtw(preds_lat, test_lat)
    dtw_lon, d_lon = sm.dtw(preds_lon, test_lon)

    # mean absolute error
    mae_lat = sm.mae(preds_lat, test_lat)
    mae_lon = sm.mae(preds_lon, test_lon)

    # mean squared error
    mse_lat = sm.mse(preds_lat, test_lat)
    mse_lon = sm.mse(preds_lon, test_lon)

    # Take the average of the metrics
    return {
        'PCM': (pcm_lat + pcm_lon) / 2,
        'DF': (df_lat + df_lon) / 2,
        'AREA': (area_lat + area_lon) / 2,
        'CL': (cl_lat + cl_lon) / 2,
        'DTW': (dtw_lat + dtw_lon) / 2,
        'MAE': (mae_lat + mae_lon) / 2,
        'MSE': (mse_lat + mse_lon) / 2
    }

def calculateMetrics(pred_dist, pred_np, y_test, y_test_ten, lat_tc, verbose=True):
    # Calculate RMSE
    rmse_0 = np.sqrt(mean_squared_error(y_test[:, 0], pred_np[:, 0]))
    rmse_1 = np.sqrt(mean_squared_error(y_test[:, 1], pred_np[:, 1]))
    rmse = np.sqrt(rmse_0**2 + rmse_1**2)

    # Calculate MAE
    mae_speed = mean_absolute_error(y_test[:, 0], pred_np[:, 0])
    mae_bearing = mean_absolute_error(y_test[:, 1], pred_np[:, 1])
    mae = np.sqrt(mae_speed**2 + mae_bearing**2)

    # Calculate MAPE
    #mape_speed = np.mean(np.abs((pred_np[:, 0] - y_test[:, 0]) / y_test[:, 0])) * 100
    #mape_bearing = np.mean(np.abs((pred_np[:, 1] - y_test[:, 1]) / y_test[:, 1])) * 100

    # Calculate NLPD
    nlpd = gpytorch.metrics.negative_log_predictive_density(pred_dist, y_test_ten)

    # Calculate MSLL
    msll = gpytorch.metrics.mean_standardized_log_loss(pred_dist, y_test_ten)
    msll = np.sqrt(msll[0].item()**2 + msll[1].item()**2)

    # Calculate similarity measures
    sim = evaluate_similarity(lat_tc, pred_np, y_test)

    if verbose:
        print('RMSE: ', rmse)
        print('MAE: ', mae)
        print('NLPD: ', nlpd)
        print('MSLL: ', msll)
        print('PCM: ', sim['PCM'] )
        print('DF: ', sim['DF'] )
        print('AREA: ', sim['AREA'] )
        print('CL: ', sim['CL'] )
        print('DTW: ', sim['DTW'] )

    return {
            'RMSE': rmse, 
            'MAE': mae, 
            'NLPD': nlpd.item(), 
            'MSLL': msll, 
            'PCM': sim['PCM'], 
            'DF': sim['DF'], 
            'AREA': sim['AREA'], 
            'CL': sim['CL'], 
            'DTW': sim['DTW']
            }

def calculate_nlpd_and_msll(mean_pred, std_dev_pred, y_true):
    """
    Calculate the Negative Log Predictive Density (NLPD) and Mean Standardized Log Loss (MSLL)
    given the mean predictions, standard deviation of predictions, and true target values.

    Parameters:
    mean_pred (np.ndarray): The mean predictions from Monte Carlo dropout inference.
    std_dev_pred (np.ndarray): The standard deviation of predictions from Monte Carlo dropout inference.
    y_true (np.ndarray): The true target values. 2xn array where n is the number of data points.

    Returns:
    tuple: A tuple containing the NLPD and MSLL values.
    """
    n = len(y_true)
    # Calculate NLPD
    # Assume a Gaussian predictive distribution
    log_likelihoods = norm.logpdf(y_true, mean_pred, std_dev_pred)
    nlpd = -np.sum(log_likelihoods) / n
    
    # Calculate MSLL
    # Standardize log loss by subtracting the baseline log loss
    # Assuming the baseline is a constant model predicting the mean of y_true
    y_mean_0 = y_true[:, 0].mean()
    y_var_0 = y_true[:, 0].var()
    y_mean_1 = y_true[:, 1].mean()
    y_var_1 = y_true[:, 1].var()
    
    # Log likelihood of a constant model predicting y_mean
    baseline_log_likelihood_0 = norm.logpdf(y_true[:, 0], y_mean_0, np.sqrt(y_var_0))
    baseline_log_loss_0 = -np.sum(baseline_log_likelihood_0) / n
    baseline_log_likelihood_1 = norm.logpdf(y_true[:, 1], y_mean_1, np.sqrt(y_var_1))
    baseline_log_loss_1 = -np.sum(baseline_log_likelihood_1) / n
    
    # Log likelihood of the predictive model
    model_log_likelihood = log_likelihoods.mean()
    
    # Calculate MSLL
    msll_0 = model_log_likelihood - baseline_log_loss_0
    msll_1 = model_log_likelihood - baseline_log_loss_1
    msll = np.sqrt(msll_0**2 + msll_1**2)
    
    return {'NLPD': nlpd, 'MSLL': msll}

def calculateMetricsAlt(pred_np, std_dev, y_test, lat_tc, verbose=True):
    # Calculate RMSE
    rmse_speed = np.sqrt(mean_squared_error(y_test[:, 0], pred_np[:, 0]))
    rmse_bearing = np.sqrt(mean_squared_error(y_test[:, 1], pred_np[:, 1]))
    rmse = np.sqrt(rmse_speed**2 + rmse_bearing**2)

    # Calculate MAE
    mae_speed = mean_absolute_error(y_test[:, 0], pred_np[:, 0])
    mae_bearing = mean_absolute_error(y_test[:, 1], pred_np[:, 1])
    mae = np.sqrt(mae_speed**2 + mae_bearing**2)

    # Calculate MAD
    mad_speed = median_absolute_error(y_test[:, 0], pred_np[:, 0])
    mad_bearing = median_absolute_error(y_test[:, 1], pred_np[:, 1])
    mad = np.sqrt(mae_speed**2 + mae_bearing**2)

    # Calculate NLPD and MSLL
    probabilistic = calculate_nlpd_and_msll(pred_np, std_dev, y_test)

    # Calculate similarity measures
    sim = evaluate_similarity(lat_tc, pred_np, y_test)

    if verbose:
        print('RMSE: ', rmse)
        print('MAE: ', mae)
        print('MAD: ', mad)
        print('NLPD: ', probabilistic['NLPD'])
        print('MSLL: ', probabilistic['MSLL'])
        print('PCM: ', sim['PCM'] )
        print('DF: ', sim['DF'] )
        print('AREA: ', sim['AREA'] )
        print('CL: ', sim['CL'] )
        print('DTW: ', sim['DTW'] )

    return {
            'RMSE': rmse,
            'MAE': mae,
            'MAD': mad,
            'NLPD': probabilistic['NLPD'],
            'MSLL': probabilistic['MSLL'],
            'PCM': sim['PCM'],
            'DF': sim['DF'],
            'AREA': sim['AREA'],
            'CL': sim['CL'],
            'DTW': sim['DTW']
            }