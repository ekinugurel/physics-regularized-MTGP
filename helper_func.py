"""
Author: Ekin Ugurel

Citation: 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt, pi, atan2, degrees
import geopandas as gpd
import skmob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

def init():
    plt.rcdefaults()
    
def haversine(lat1, long1, lat2, long2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees); in meters

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    long1 : float
        Longitude of the first point.
    lat2 : float
        Latitude of the second point.
    long2 : float
        Longitude of the second point.
    """
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlong = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2
    c = 2 * asin(sqrt(a)) 
    R = 6371  # radius of the earth in km
    m = R * c * 1000
    return m

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees); in meters

    Parameters
    ----------
    lon1 : float
        Longitude of the first point.
    lat1 : float
        Latitude of the first point.
    lon2 : float
        Longitude of the second point.
    lat2 : float
        Latitude of the second point.
    """
    # Convert latitude and longitude to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Calculate the difference between latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r * 1000 # Output distance in meters

def geodesic(lat1, lon1, lat2, lon2):
    """
    Geodesic distance; in meters

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    lon1 : float
        Longitude of the first point.
    lat2 : float
        Latitude of the second point.
    lon2 : float
        Longitude of the second point.
    """
    
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
    R = 6371  # radius of the earth in km
    x = (lon2 - lon1) * cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    d = R * sqrt( x*x + y*y ) * 1000
    return d

def addBearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two points

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    lon1 : float
        Longitude of the first point.
    lat2 : float
        Latitude of the second point.
    lon2 : float
        Longitude of the second point.
    """
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))
    dLon = lon2 - lon1
    y = sin(dLon) * cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng

def addBearingAlt(lat, lon, verbose=False):
    """
    Calculates the bearing between two points

    Parameters
    ----------
    lat: 
        column containing latitudes
    lon:
        column containing longitudes
    verbose:
        boolean, whether to print out the progress of the function
    """
    if verbose:
        print("Adding bearing column to dataframe...")
    lat1, lon1 = lat, lon
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    bearing.iloc[-1] = 0
    return bearing

def newCoords(lat, lon, dy, dx):
    """
    Calculates a new lat/lon from an old lat/lon + displacement in x and y.

    Parameters
    ----------
    lat : float
        Latitude of the first point.
    lon : float
        Longitude of the first point.
    dy : float
        Displacement in y.
    dx : float
        Displacement in x.
    """
    r = 6371
    new_lat  = lat  + (dy*0.001 / r) * (180 / pi)
    new_lon = lon + (dx*0.001 / r) * (180 / pi) / cos(lat * pi/180)
    return new_lat, new_lon

def newCoordsAlt(lat1, lon1, d, brng, R = 6378.1):
    """
    Calculates a new lat/lon from an old lat/lon + distance and bearing

    Parameters
    ----------
    lat1 : float
        Latitude of the first point.
    lon1 : float
        Longitude of the first point.
    d : float
        Distance between the two points.
    brng : float
        Bearing between the two points.
    R : float
        Radius of the earth. Default is 6378.1 km.
    """
    d = d/1000 # convert distance to km
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    
    lat2 = asin( sin(lat1)*cos(d/R) +
         cos(lat1)*sin(d/R)*cos(brng))
    
    lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),
                 cos(d/R)-sin(lat1)*sin(lat2))
    
    lat2 = degrees(lat2)
    lon2 = degrees(lon2)
    return lat2, lon2

def addDist(data, type=haversine_np, lat='orig_lat', lon='orig_long', verbose=False):
    """
    Add distance column to a dataframe with latitudes and longitudes. 
    Type specifies whether to use the Haversine distance (default) or the geodesic distance. 
    Returned distance is in meters.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe with latitudes and longitudes.
    type : function
        Function to calculate distance. Default is haversine_np.
    lat : string
        Name of the column with latitudes. Default is 'orig_lat'.
    lon : string
        Name of the column with longitudes. Default is 'orig_long'.
    """
    if verbose:
        print("Adding distance column to dataframe...")
    if type == haversine_np:
        # Calculate distance between each point and the next point
        lat1, lon1 = data[lat], data[lon]
        # Shift lat/lon columns by 1 row to get the next point
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type(lat1, lon1, lat2, lon2).fillna(0)
        data['dist'] = dist
    elif type == geodesic:
        lat1, lon1 = data[lat], data[lon]
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = type(lat1, lon1, lat2, lon2).miles.fillna(0)
        data['dist'] = dist
        
def addVel(data, unix='unix_min', lat='orig_lat', lon='orig_long', verbose=False):
    """
    Add velocity column to a dataframe with latitudes and longitudes. 
    Speed in meters/second.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe with latitudes and longitudes.
    unix : string
        Name of the column containing unix timestamps.
    lat : string
        Name of the column containing latitudes.
    lon : string
        Name of the column containing longitudes.
    """
    if verbose:
        print("Adding velocity column to dataframe...")
    if 'dist' in data.columns:
        lat1, lon1 = data[lat], data[lon]
        lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
        dist = data['dist'].fillna(0)
        time_diff = (data[unix] - data[unix].shift(1)).fillna(0)
        vel = dist / time_diff
        vel.iloc[0] = 0
        vel.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinite values with NaN
        data['vel'] = vel
    else:
        print("Please run addDist method to calculate distances between points first.")

def preds_to_full_df(preds_lat, preds_long, test_df, train_df, 
                     unix='unix_min', datetime='date', lat='lat', long='long'):
    '''
    Function to merge the predictions with the original training set to create a full dataframe.
    
    '''
    # Create dataframe with GP predictions
    orig_preds_df = pd.DataFrame(test_df[unix], columns=[unix])
    orig_preds_df[datetime] = test_df[datetime]
    orig_preds_df[lat] = preds_lat
    orig_preds_df[long] = preds_long

    tdf = pd.concat([train_df, orig_preds_df], axis=0)

    # Sort by unix time
    tdf.sort_values(by=unix, inplace=True)

    # Rename datetime column to datetime
    tdf.rename(columns={datetime: 'datetime'}, inplace=True)

    return tdf

def uniqueid():
    seed = random.getrandbits(32)
    while True:
        yield seed
        seed += 1

def dec_floor(a, precision=1):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)
    
def min_float(a, b):
    """
    Returns the minimum of two floats.

    Parameters
    ----------
    a : float
        First float.
    b : float
        Second float.

    Returns
    -------
    float
        The minimum of the two floats.
    """
    if a < b:
        return a
    else:
        return b
    
def max_float(a, b):
    """
    Returns the maximum of two floats.

    Parameters
    ----------
    a : float
        First float.
    b : float
        Second float.

    Returns
    -------
    float
        The maximum of the two floats.
    """
    if a > b:
        return a
    else:
        return b
    
def spatialKernelPredPlots(X_test_spat, pred_dist, mean, y_test, y_test_tens, verbose=True):
    # Obtain mean
    #mean = pred_dist.mean
    # Plot predictions for both speed and bearing, and compare with ground truth
    # Make colorbar have same range for all plots, 95% percentile of all values
    vmin_vel = min_float(np.percentile(mean[:, 0], 5), np.percentile(y_test[:, 0], 5))
    vmax_vel = max_float(np.percentile(mean[:, 0], 95), np.percentile(y_test[:, 0], 95))
    vmin_bear = min_float(np.percentile(mean[:, 1], 5), np.percentile(y_test[:, 1], 5))
    vmax_bear = max_float(np.percentile(mean[:, 1], 95), np.percentile(y_test[:, 1], 95))
    plt.subplots(2, 2, figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.scatter(X_test_spat[:, 1], X_test_spat[:, 0], c=mean[:, 0], cmap='coolwarm')
    plt.clim(vmin=vmin_vel, vmax=vmax_vel)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Predicted speed')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.scatter(X_test_spat[:, 1], X_test_spat[:, 0], c=y_test[:, 0], cmap='coolwarm')
    plt.clim(vmin=vmin_vel, vmax=vmax_vel)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Ground truth speed')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.scatter(X_test_spat[:, 1], X_test_spat[:, 0], c=mean[:, 1], cmap='coolwarm')
    plt.clim(vmin=vmin_bear, vmax=vmax_bear)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Predicted bearing')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.scatter(X_test_spat[:, 1], X_test_spat[:, 0], c=y_test[:, 1], cmap='coolwarm')
    plt.clim(vmin=vmin_bear, vmax=vmax_bear)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title('Ground truth bearing')
    plt.colorbar()
    plt.show()

    # Measure RMSE, not with torch tensors
    rmse_speed = np.sqrt(mean_squared_error(y_test[:, 0], mean[:, 0]))

    # Measure MAE
    mae_speed = mean_absolute_error(y_test[:, 0], mean[:, 0])

    # Measure MAPE
    #mape_speed = np.mean(np.abs((mean[:, 0] - y_test[:, 0]) / y_test[:, 0]))

    # Measure RMSE
    rmse_bearing = np.sqrt(mean_squared_error(y_test[:, 1], mean[:, 1]))

    # Measure MAE
    mae_bearing = mean_absolute_error(y_test[:, 1], mean[:, 1])

    # Measure MAPE
    #mape_bearing = np.mean(np.abs((mean[:, 1] - y_test[:, 1]) / y_test[:, 1]))

    # Calculate NLPD
    nlpd = gpytorch.metrics.negative_log_predictive_density(pred_dist, y_test_tens)

    # Calculate MSLL
    msll = gpytorch.metrics.mean_standardized_log_loss(pred_dist, y_test_tens)

    if verbose:
        print('RMSE speed: ', rmse_speed)
        print('MAE speed: ', mae_speed)
        #print('MAPE speed: ', mape_speed)
        print('RMSE bearing: ', rmse_bearing)
        print('MAE bearing: ', mae_bearing)
        #print('MAPE bearing: ', mape_bearing)
        print('NLPD: ', nlpd)
        print('MSLL: ', msll)

    return rmse_speed, mae_speed, rmse_bearing, mae_bearing, nlpd, msll

def temporalKernelPredPlots(X_test_temp, pred_dist, mean, y_test, y_test_tens, plot_ver=0, verbose=True):
    if plot_ver == 0:
        # Plot predictions
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(y_test[:,0], mean[:,0], 'o')
        axs[0, 0].plot(y_test[:,0], y_test[:,0], 'k-')
        axs[0, 0].set_xlabel('True speed')
        axs[0, 0].set_ylabel('Predicted speed')
        axs[0, 0].set_title('Speed')
        axs[0, 1].plot(y_test[:,1], mean[:,1], 'o')
        axs[0, 1].plot(y_test[:,1], y_test[:,1], 'k-')
        axs[0, 1].set_xlabel('True bearing')
        axs[0, 1].set_ylabel('Predicted bearing')
        axs[0, 1].set_title('Bearing')
        axs[1, 0].plot(y_test[:,0], mean[:,0] - y_test[:,0], 'o')
        axs[1, 0].set_xlabel('True speed')
        axs[1, 0].set_ylabel('Residuals')
        axs[1, 0].set_title('Speed')
        axs[1, 1].plot(y_test[:,1], mean[:,1] - y_test[:,1], 'o')
        axs[1, 1].set_xlabel('True bearing')
        axs[1, 1].set_ylabel('Residuals')
        axs[1, 1].set_title('Bearing')
        # Legend
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif plot_ver == 1:
    # Plot predictions over time
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(X_test_temp[:,0], mean[:,0], 'o', label='Predicted')
        axs[0, 0].plot(X_test_temp[:,0], y_test[:,0], 'k-', label='True')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Speed')

        axs[0, 1].plot(X_test_temp[:,0], mean[:,1], 'o', label='Predicted')
        axs[0, 1].plot(X_test_temp[:,0], y_test[:,1], 'k-', label='True')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Bearing')

        axs[1, 0].plot(X_test_temp[:,0], mean[:,0] - y_test[:,0], 'o')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Residuals')

        axs[1, 1].plot(X_test_temp[:,0], mean[:,1] - y_test[:,1], 'o')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Residuals')

        # Legend
        axs[0, 0].legend()        

        plt.tight_layout()
        plt.show()


    # Calculate RMSE
    temp_rmse_speed = np.sqrt(mean_squared_error(y_test[:, 0], mean[:, 0]))
    temp_rmse_bearing = np.sqrt(mean_squared_error(y_test[:, 1], mean[:, 1]))

    # Calculate MAE
    temp_mae_speed = mean_absolute_error(y_test[:, 0], mean[:, 0])
    temp_mae_bearing = mean_absolute_error(y_test[:, 1], mean[:, 1])

    # Calculate MAPE
    #temp_mape_speed = np.mean(np.abs((mean[:, 0] - y_test[:, 0]) / y_test[:, 0])) * 100
    #temp_mape_bearing = np.mean(np.abs((mean[:, 1] - y_test[:, 1]) / y_test[:, 1])) * 100

    # Calculate NLPD
    temp_nlpd = gpytorch.metrics.negative_log_predictive_density(pred_dist, y_test_tens)

    # Calculate MSLL
    temp_msll = gpytorch.metrics.mean_standardized_log_loss(pred_dist, y_test_tens)

    if verbose:
        print('RMSE speed: ', temp_rmse_speed)
        print('MAE speed: ', temp_mae_speed)
        #print('MAPE speed: ', temp_mape_speed)
        print('RMSE bearing: ', temp_rmse_bearing)
        print('MAE bearing: ', temp_mae_bearing)
       #print('MAPE bearing: ', temp_mape_bearing)
        print('NLPD: ', temp_nlpd)
        print('MSLL: ', temp_msll)
    
    return temp_rmse_speed, temp_mae_speed, temp_rmse_bearing, temp_mae_bearing, temp_nlpd, temp_msll

def normalizeLatLng(train, test, lat_col = 'lat', lng_col = 'lng'):
    """
    Normalize latitude and longitude columns in a dataframe.

    Parameters
    ----------
    train : pandas dataframe
        Dataframe containing training data.
    test : pandas dataframe
        Dataframe containing test data.
    lat_col : string
        Name of the column containing latitude. Default is 'lat'.
    lng_col : string
        Name of the column containing longitude. Default is 'lng'.
    """
    scaler_lat = StandardScaler()
    scaler_lng = StandardScaler()
    train['norm_lat'] = scaler_lat.fit_transform(train[lat_col].to_numpy().reshape(-1, 1))
    train['norm_lng'] = scaler_lng.fit_transform(train[lng_col].to_numpy().reshape(-1, 1))
    test['norm_lat'] = scaler_lat.transform(test[lat_col].to_numpy().reshape(-1, 1))
    test['norm_lng'] = scaler_lng.transform(test[lng_col].to_numpy().reshape(-1, 1))
    return scaler_lat, scaler_lng

def normalizePhys(train, test, speed_col='speed', bearing_col='bearing'):
    """
    Normalize speed and bearing columns in a dataframe.

    Parameters
    ----------
    train : pandas dataframe
        Dataframe containing training data.
    test : pandas dataframe
        Dataframe containing test data.
    """
    scaler_speed = StandardScaler()
    scaler_bearing = StandardScaler()
    train['norm_speed'] = scaler_speed.fit_transform(train[speed_col].to_numpy().reshape(-1, 1))
    train['norm_bearing'] = scaler_bearing.fit_transform(train[bearing_col].to_numpy().reshape(-1, 1))
    test['norm_speed'] = scaler_speed.transform(test[speed_col].to_numpy().reshape(-1, 1))
    test['norm_bearing'] = scaler_bearing.transform(test[bearing_col].to_numpy().reshape(-1, 1))
    return scaler_speed, scaler_bearing

def normalizeUnix(SPTVars):
    """
    Normalize unix time columns in a dataframe.
    """
    train_unix = SPTVars.train_temp[:, 0]
    test_unix = SPTVars.test_temp[:, 0]

    SPTVars.train_temp[:, 0] = torch.nn.functional.normalize(train_unix, dim=-1)
    SPTVars.test_temp[:, 0] = torch.nn.functional.normalize(test_unix, dim=-1)

def reverseStandardizePhys(scaler_speed, scaler_bearing, preds, y_test_phys, lower, upper):
    preds_speed = scaler_speed.inverse_transform(preds[:, 0].numpy().reshape(-1, 1))
    preds_bearing = scaler_bearing.inverse_transform(preds[:, 1].numpy().reshape(-1, 1))
    y_test_speed = scaler_speed.inverse_transform(y_test_phys[:, 0].numpy().reshape(-1, 1))
    y_test_bearing = scaler_bearing.inverse_transform(y_test_phys[:, 1].numpy().reshape(-1, 1))

    lower_speed = scaler_speed.inverse_transform(lower[:, 0].numpy().reshape(-1, 1))
    lower_bearing = scaler_bearing.inverse_transform(lower[:, 1].numpy().reshape(-1, 1))
    upper_speed = scaler_speed.inverse_transform(upper[:, 0].numpy().reshape(-1, 1))
    upper_bearing = scaler_bearing.inverse_transform(upper[:, 1].numpy().reshape(-1, 1))
    
    # Concatenate arrays
    pred_np = np.concatenate((preds_speed, preds_bearing), axis=1)
    y_test_np = np.concatenate((y_test_speed, y_test_bearing), axis=1)
    lower = np.concatenate((lower_speed, lower_bearing), axis=1)
    upper = np.concatenate((upper_speed, upper_bearing), axis=1)

    return pred_np, y_test_np, lower, upper

def reverseStandardizeSpat(scaler_lat, scaler_lng, preds, y_test_spat, lower, upper):
    preds_lat = scaler_lat.inverse_transform(preds[:, 0].numpy().reshape(-1, 1))
    preds_lng = scaler_lng.inverse_transform(preds[:, 1].numpy().reshape(-1, 1))
    y_test_lat = scaler_lat.inverse_transform(y_test_spat[:, 0].numpy().reshape(-1, 1))
    y_test_lng = scaler_lng.inverse_transform(y_test_spat[:, 1].numpy().reshape(-1, 1))

    lower_lat = scaler_lat.inverse_transform(lower[:, 0].numpy().reshape(-1, 1))
    lower_lng = scaler_lng.inverse_transform(lower[:, 1].numpy().reshape(-1, 1))
    upper_lat = scaler_lat.inverse_transform(upper[:, 0].numpy().reshape(-1, 1))
    upper_lng = scaler_lng.inverse_transform(upper[:, 1].numpy().reshape(-1, 1))

    # Concatenate arrays
    pred_np = np.concatenate((preds_lat, preds_lng), axis=1)
    y_test_np = np.concatenate((y_test_lat, y_test_lng), axis=1)
    lower = np.concatenate((lower_lat, lower_lng), axis=1)
    upper = np.concatenate((upper_lat, upper_lng), axis=1)

    return pred_np, y_test_np, lower, upper

def plot_errors_pts(X, lat_rmse, lng_rmse, lat_mae, lng_mae, nlpd, train_times, pred_times, fig_size=(10, 5)):
    X_axis = np.arange(len(X))
    fig, axs = plt.subplots(1, 4, figsize=fig_size)
    axs[0].bar(X_axis - 0.2, lat_rmse, 0.4, label = 'Lat')
    axs[0].bar(X_axis + 0.2, lng_rmse, 0.4, label = 'Lng')

    axs[0].set_xticks(X_axis)
    axs[0].set_xticklabels(X)
    axs[0].set_xlabel("Model")
    axs[0].set_ylabel("RMSE")
    axs[0].legend()

    axs[1].bar(X_axis - 0.2, lat_mae, 0.4, label = 'Lat')
    axs[1].bar(X_axis + 0.2, lng_mae, 0.4, label = 'Lng')

    axs[1].set_xticks(X_axis)
    axs[1].set_xticklabels(X)
    axs[1].set_xlabel("Model")
    axs[1].set_ylabel("MAE")
    axs[1].legend()

    axs[2].bar(X_axis, nlpd, label = 'Lat', color='brown')

    axs[2].set_xticks(X_axis)
    axs[2].set_xticklabels(X)
    axs[2].set_xlabel("Model")
    axs[2].set_ylabel("NLPD")

    axs[3].bar(X_axis - 0.2, train_times, 0.4, label = 'Train', color='green')
    axs[3].bar(X_axis + 0.2, pred_times, 0.4, label = 'Predict', color='red')

    axs[3].set_xticks(X_axis)
    axs[3].set_xticklabels(X)
    axs[3].set_xlabel("Model")
    axs[3].set_ylabel("(Log) Seconds")
    # Put it in the lower left
    axs[3].legend(loc='lower left')

    plt.tight_layout()
    plt.show()

def plot_times(train_times, pred_times, X):
    X_axis = np.arange(len(X))
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.bar(X_axis - 0.2, train_times, 0.4, label = 'Train')
    axs.bar(X_axis + 0.2, pred_times, 0.4, label = 'Predict')

    axs.set_xticks(X_axis)
    axs.set_xticklabels(X)
    axs.set_xlabel("Model")
    axs.set_ylabel("(Log) Seconds")
    axs.legend()

    plt.tight_layout()
    plt.show()

def plot_lat_lng(df_train, df_test, full_mean_rev_kron, full_lower_rev_kron, full_upper_rev_kron):
    # Plot lat/lng as a function of time, scatter plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    ax[0].scatter(pd.to_datetime(df_train['datetime']), df_train['norm_lat'], label='Train', s=1, c='b')
    ax[0].scatter(pd.to_datetime(df_test['datetime']), df_test['norm_lat'], label='Test', s=1, c='green')
    ax[0].scatter(pd.to_datetime(df_test['datetime']), full_mean_rev_kron[:, 0], label='Predicted', s=1, c='red')
    ax[0].fill_between(pd.to_datetime(df_test['datetime']), full_lower_rev_kron[:,0], full_upper_rev_kron[:,0], alpha=0.2, color='red')
    ax[0].set_ylabel('Latitude')
    ax[0].set_xticklabels([])

    ax[1].scatter(pd.to_datetime(df_train['datetime']), df_train['norm_lng'], label='Train', s=1, c='b')
    ax[1].scatter(pd.to_datetime(df_test['datetime']), df_test['norm_lng'], label='Test', s=1, c='green')
    ax[1].scatter(pd.to_datetime(df_test['datetime']), full_mean_rev_kron[:, 1], label='Predicted', s=1, c='red')
    ax[1].fill_between(pd.to_datetime(df_test['datetime']), full_lower_rev_kron[:,1], full_upper_rev_kron[:,1], alpha=0.2, color='red')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Longitude')
    ax[1].tick_params(axis='x', rotation=20)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def gpd_physics_cols(df, projection='EPSG:32610', xy_separation=False, centroid_cols=False):
    # Load into geopandas dataframe
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.orig_long, df.orig_lat), crs="EPSG:4326")

    # Project to UTM Zone 10N
    gdf = gdf.to_crs(projection)
    
    # Insert column of distances between consecutive points
    gdf['dist'] = gdf['geometry'].distance(gdf['geometry'].shift())

    if xy_separation:
        # Insert columns of x and y distances between consecutive points
        gdf['x_dist'] = gdf['geometry'].x - gdf['geometry'].shift().x
        gdf['y_dist'] = gdf['geometry'].y - gdf['geometry'].shift().y

    # Insert column of time differences between consecutive points
    gdf['time_diff'] = gdf['unix_start_t'] - gdf['unix_start_t'].shift()

    # Insert column of speeds between consecutive points
    gdf['speed'] = gdf['dist'] / gdf['time_diff']

    if xy_separation:
        # Insert columns of x and y speeds between consecutive points
        gdf['x_speed'] = gdf['x_dist'] / gdf['time_diff']
        gdf['y_speed'] = gdf['y_dist'] / gdf['time_diff']

    # Replace NaN values with 0
    gdf = gdf.fillna(0)

    if centroid_cols:
        # Find centroid of all points, then derive x and y distances from centroid
        centroid = gdf.dissolve().centroid
        gdf['x_dist_from_centroid'] = gdf['geometry'].x - centroid.x[0]
        gdf['y_dist_from_centroid'] = gdf['geometry'].y - centroid.y[0]

    # Add bearing column
    df['bearing'] = addBearingAlt(df['orig_lat'], df['orig_long'])

    return gdf

def add_temporal_cols(gdf, unix_col = 'unix'):
    # Read as dataframe 
    df = pd.DataFrame(gdf)

    # Add hours column
    df['hours'] = df['datetime'].dt.hour

    # Add seconds after midnight column
    df['seconds_after_midnight'] = df['hours'] * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second

    # Add day of week column
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Cyclical encoding using sine and cosine functions
    df['hour_sin'] = np.sin(2 * np.pi * df['hours'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hours'] / 24)
    df['sam_sin'] = np.sin(2 * np.pi * df['seconds_after_midnight'] / 86400)
    df['sam_cos'] = np.cos(2 * np.pi * df['seconds_after_midnight'] / 86400)

    # Convert into one-hot encoding
    df = pd.get_dummies(df, columns=['day_of_week'])

    # If column exists, cast as int
    if 'day_of_week_0' in df.columns:
        df['day_of_week_0'] = df['day_of_week_0'].astype(int)
    if 'day_of_week_1' in df.columns:
        df['day_of_week_1'] = df['day_of_week_1'].astype(int)
    if 'day_of_week_2' in df.columns:
        df['day_of_week_2'] = df['day_of_week_2'].astype(int)
    if 'day_of_week_3' in df.columns:
        df['day_of_week_3'] = df['day_of_week_3'].astype(int)
    if 'day_of_week_4' in df.columns:
        df['day_of_week_4'] = df['day_of_week_4'].astype(int)
    if 'day_of_week_5' in df.columns:
        df['day_of_week_5'] = df['day_of_week_5'].astype(int)
    if 'day_of_week_6' in df.columns:
        df['day_of_week_6'] = df['day_of_week_6'].astype(int)

    # Standardize unix_start_t such that the first value is 0
    df[unix_col] = df[unix_col] - df[unix_col].iloc[0]

    return df
    
def hcfnaive(a,b):
    if b==0:
        return a
    else:
        return hcfnaive(b,a%b)