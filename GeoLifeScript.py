#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import utils, methods
import torch
import skmob
from sklearn.cluster import KMeans
import plots
import warnings
import models
import numpy as np
import helper_func as hf
import metrics
import torch.nn as nn
import torch.optim as optim
import os
import benchmarkMethods as BM
import time as tm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.kernels import ScaleKernel, RQKernel as RQ, MaternKernel as MAT
import MKL
import argparse
warnings.filterwarnings("ignore")

def main(input_path_traj, 
         input_path_comp,
         output_path, 
         num_epochs=50, 
         num_lstm_runs=70,
         GP_learning_rate=0.1,
         LSTM_learning_rate=0.01, 
         num_inducing=200, 
         compression_radius=0.2, 
         max_speed_filter=300, 
         num_latents=3, 
         n_MKL_epochs=20,
         max_depth = 3,
         hidden_size=48, 
         num_layers=2, 
         output_size=2, 
         batch_size=16,
         seed=42,
         verbose=False):
    """
    Main function to run the GeoLife script.

    Parameters
    ----------
    input_path_traj : str
        Path to the input folder containing trajectory data.
    input_path_comp : str
        Path to the input folder containing compressed trip data.
    output_path : str
        Path to the output data.
    num_epochs : int
        Number of epochs to train the model.
    num_lstm_runs : int
        Number of runs for the LSTM model.
    GP_learning_rate : float
        Learning rate for the GP model.
    LSTM_learning_rate : float
        Learning rate for the LSTM model.
    num_inducing : int
        Number of inducing points for the GP model.
    compression_radius : float
        Radius for compressing the trajectory data.
    max_speed_filter : int
        Maximum speed for filtering the data.
    num_latents : int
        Number of latents for the GP model.
    n_MKL_epochs : int  
        Number of epochs for the MKL model.
    max_depth : int
        Maximum depth for the MKL tree.
    hidden_size : int
        Hidden size for the LSTM model.
    num_layers : int
        Number of layers for the LSTM model.
    output_size : int
        Output size for the LSTM model.
    batch_size : int
        Batch size for the LSTM model.
    seed : int
        Seed for the random number generator.
    verbose : bool
        Whether to print verbose output.
    """
    
    os.chdir(input_path_traj)
    
    # Read compressed data
    compressed = pd.read_csv(input_path_comp + '/full_geolife+weather.csv')

    # Remove '.plt' from Id_perc
    compressed['Id_perc'] = compressed['Id_perc'].replace('.plt', '', regex=True).astype(float) 

    # Read the folder names in the data folder
    ids = utils.read_folder_names(input_path_traj) 

    # Set seed
    torch.manual_seed(seed)

    # Display all modes
    labels = ['walk', 'bus', 'bike']
    lstm = True
    GP = True
    sparseGP = True

    for id in ids:
        user_path = output_path + '/user_{}'.format(id)
        user_path_results = output_path + '/all_results/'
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        print("Starting tests on User", id)
        for label in labels:
            if not os.path.exists(user_path + '/' + label):
                os.makedirs(user_path + '/' + label)

            # Check if the CSV file already exists
            csv_filename = 'metrics_{}_{}.csv'.format(id, label)
            if os.path.exists(os.path.join(user_path_results + csv_filename)):
                print("CSV file already exists for User", id, "and Mode", label)
                continue
                
            print(f"User {id} has {compressed[(compressed['Id_user'] == id) & (compressed['label'] == label)].shape[0]} trips with mode {label}")
            trips = compressed[(compressed['Id_user'] == id) & (compressed['label'] == label)]

            # If the number of trips with this mode is less than 10, skip to the next mode
            if trips.shape[0] < 10:
                print("Too few trips for this mode, skipping to the next mode...")
                continue
            
            # Read the trajectory data for the user
            traj_data = utils.read_traj_data(input_path_traj, id)
            traj_data['datetime'] = pd.to_datetime(traj_data['date'] + ' ' + traj_data['time'])
            traj_data = traj_data.sort_values(by='datetime')
            # Use the Id_perc to get the points from tdf
            points = traj_data[traj_data['trip_id'].isin(trips['Id_perc'])]

            os.chdir(user_path + '/' + label)

            # Join some of the columns from the compressed data
            points_m = points.merge(trips[['Id_perc', 'label', 'latitudeStart', 'longitudeStart', 'latitudeEnd', 'longitudeEnd', 
                                            'TimeStart', 'TimeEnd', 'StartDay', 'EndDay', 'StartHour', 'EndHour', 'distanceTotal', 
                                            'time_total', 'npoints', 'vel_avg', 'vel_max', 'vcr', 'sr', 'hcr']], 
                                            left_on='trip_id', right_on='Id_perc', how='left')

            # Eliminate trips that start and end outside of Beijing
            points_m = points_m[(points_m['longitudeStart'] > 115) 
                                & (points_m['longitudeStart'] < 118) 
                                & (points_m['latitudeStart'] > 39) 
                                & (points_m['latitudeStart'] < 41) 
                                & (points_m['longitudeEnd'] > 115) 
                                & (points_m['longitudeEnd'] < 118) 
                                & (points_m['latitudeEnd'] > 39)
                                & (points_m['latitudeEnd'] < 41)]

            # Also eliminate trips that go south of long = 115, west of lat = 39, north of lat = 41, and east of long = 118
            # To do this, eliminate trips completely (take out all points of a tid which has a point south of long = 115)
            points_m = points_m[~points_m['trip_id'].isin(points_m[(points_m['lon'] < 115)]['trip_id'])]
            points_m = points_m[~points_m['trip_id'].isin(points_m[(points_m['lat'] < 39)]['trip_id'])]
            points_m = points_m[~points_m['trip_id'].isin(points_m[(points_m['lat'] > 41)]['trip_id'])]
            points_m = points_m[~points_m['trip_id'].isin(points_m[(points_m['lon'] > 118)]['trip_id'])]

            print('Number of trips after removing trips outside of Beijing:', points_m['trip_id'].nunique())

            if points_m['trip_id'].nunique() < 10:
                print("Too few trips for this mode, skipping to the next mode...")
                continue
            
            # Make TrajDataFrame
            tdf = skmob.TrajDataFrame(points_m, latitude='lat', longitude='lon', trajectory_id='trip_id', datetime='datetime', timestamp=True)
            # Eliminate unrealistic points (speed > 300 km/h)
            tdf_f = skmob.preprocessing.filtering.filter(tdf, max_speed_kmh=max_speed_filter, include_loops=True)
            # What percent of the data did we remove?
            percent_filtered = (len(tdf) - len(tdf_f)) / len(tdf) * 100
            print('We removed {} percent of the points by filtering'.format(percent_filtered))
            # Compress trajectory
            tdf_f_c = skmob.preprocessing.compression.compress(tdf_f, spatial_radius_km=compression_radius)
            percent_compressed = (len(tdf_f) - len(tdf_f_c)) / len(tdf_f) * 100
            points_m = pd.DataFrame(tdf_f_c)
            # How many did we remove?
            print('We further compressed the data by {} percent'.format(percent_compressed))

            # Adjust 'trips' to only include the trips that are in 'points_m'
            trips = trips[trips['Id_perc'].isin(points_m['tid'])]

            # Cluster based on the start and end points
            start = trips[['latitudeStart', 'longitudeStart']]
            end = trips[['latitudeEnd', 'longitudeEnd']]
            data = pd.concat([start, end], axis=1)

            if (points_m['tid'].nunique() >= 10) and (points_m['tid'].nunique() <= 20):
                points_m_c = points_m
                pass
            else:
                # Find the optimal number of clusters for the data
                max_clusters = 10
                optimal_clusters = methods.find_optimal_trip_clusters(data, max_clusters)
                if optimal_clusters == 1:
                    optimal_clusters = 2
                print('Optimal number of clusters for the data:', optimal_clusters)

                # Fit K-means clustering model with the optimal number of clusters
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
                data['cluster'] = kmeans.fit_predict(data)

                # Retain only the top 1 cluster
                top_clusters = data['cluster'].value_counts().head(1).index
                top_data = data[data['cluster'].isin(top_clusters)]

                # Retrieve the trips where top_data['latitudeStart'] and top_data['longitudeStart'] are the same as points_m['latitudeStart'] and points_m['longitudeStart']
                # Do the same for the end points
                points_m_c = points_m[(points_m['latitudeStart'].isin(top_data['latitudeStart'])) &\
                                    (points_m['longitudeStart'].isin(top_data['longitudeStart'])) &\
                                        (points_m['latitudeEnd'].isin(top_data['latitudeEnd'])) & \
                                            (points_m['longitudeEnd'].isin(top_data['longitudeEnd']))]

                print('Number of trips after clustering:', points_m_c['tid'].nunique())

            points_m_c_s = utils.process_data(points_m_c = points_m_c, data=data, points_m=points_m)

            print('Number of trips after sampling:', points_m_c_s['tid'].nunique())
        
            fig1 = plots.plotTrip(points_m_c_s, points_m_c_s['tid'].unique()[0], title='Similar {} for user {}'.format(label, id))
            fig1.savefig('similar_{}_user_{}.png'.format(label, id))
    
            # Add a unix column
            points_m_c_s['unix'] = (points_m_c_s['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

            # Remove duplicates by unix
            points_m_c_s = points_m_c_s.drop_duplicates(subset=['unix'], keep='first')

            # Add physical variables
            points_m_c_s = points_m_c_s.groupby('tid').apply(methods.addDist)

            points_m_c_s = points_m_c_s.groupby('tid').apply(methods.addVel)

            points_m_c_s = points_m_c_s.groupby('tid').apply(methods.addBearing)

            # Add temporal variables
            points_m_c_s = hf.add_temporal_cols(points_m_c_s)

            metrics_df = pd.DataFrame()

            k_folds_data = utils.tripLabelBasedKFoldSplit(points_m_c_s, k=3, random_state=seed)

            for i, (train, test) in enumerate(k_folds_data):
                print('Fold', i)
                # Number of trips in train and test
                print('Number of Training Trips:', train['tid'].nunique())
                print('Number of Testing Trips:', test['tid'].nunique())
                print("Number of points in training set: {}".format(len(train.iloc[:,0])))
                print("Number of points in test set: {}".format(len(test.iloc[:,0])))
                print()

                if len(train.iloc[:,0]) > 2000:
                    print("Too many points in the training set, skipping to the next fold...")
                    continue
                elif len(train.iloc[:,0]) < 100:
                    print("Too few points in the training set, skipping to the next fold...")
                    continue

                # Save train and test data
                train.to_csv('train_fold_{}.csv'.format(i), index=False)
                test.to_csv('test_fold_{}.csv'.format(i), index=False)

                # Standardize lat and lng using StandardScaler, do it separately
                scaler_lat, scaler_lng = hf.normalizeLatLng(train, test, lat_col='lat', lng_col='lng')

                # Standardize speed and bearing using StandardScaler
                scaler_speed, scaler_bearing = hf.normalizePhys(train, test, speed_col='vel', bearing_col='bearing')

                # Initialize model variables
                SPTVars = models.SPTVars(train, test, scaler_speed, scaler_bearing, scaler_lat, scaler_lng, unix_col='unix', CUDA=False)

                # Normalize Unix time
                hf.normalizeUnix(SPTVars)

                # Unix time for benchmarks
                unix_min_tr = np.array(train['unix']).astype(int)
                unix_min_te = np.array(test['unix']).astype(int)

                lat, lat_tc, lon, lon_tc = utils.makeSeries(SPTVars.train_spat, SPTVars.test_spat, unix_min_tr, unix_min_te)

                if GP: 
                    try:
                        # Operations allowed within MKL
                        algebra = {'+': lambda x, y: x + y,
                                '*': lambda x, y: x * y
                                }

                        
                        mat32_kern = ScaleKernel(MAT(nu=1.5, batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents]))
                        rq_kern = ScaleKernel(RQ(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents]))
                        
                        kernels_list = [mat32_kern, rq_kern]

                        # MKL
                        GK = MKL.GreedyKernel(algebra, kernels_list, n_epochs=n_MKL_epochs, sparse=True)
                        GK.grow_tree(SPTVars.train_temp, SPTVars.train_spat, max_depth=max_depth)

                        if (GK.str_kernel == 'MaternKernel * RQKernel') or (GK.str_kernel == 'RQKernel * MaternKernel'):
                            temp_kernel = ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1])) * \
                                            ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1]))
                        elif (GK.str_kernel == 'RQKernel + MaternKernel') or (GK.str_kernel == 'MaternKernel + RQKernel'):
                            temp_kernel = ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1])) + \
                                            ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1]))
                        elif (GK.str_kernel == 'RQKernel * RQKernel'):
                            temp_kernel = ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1])) * \
                                            ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1]))
                        elif (GK.str_kernel == 'RQKernel + RQKernel'):
                            temp_kernel = ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1])) + \
                                            ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1]))
                        elif (GK.str_kernel == 'MaternKernel + MaternKernel'):
                            temp_kernel = ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1])) + \
                                            ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1]))
                        elif (GK.str_kernel == 'MaternKernel * MaternKernel'):
                            temp_kernel = ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1])) * \
                                            ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1]))
                        else:
                            temp_kernel = ScaleKernel(MAT(nu=1.5, ard_num_dims=SPTVars.train_temp.shape[1])) * \
                                            ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1]))
                        # PIMTGP Model
                        temp_kernel = ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1])) * \
                                                ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1]))
                        full_kernel =   ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                        ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) +\
                                        ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                        ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) * \
                                        ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1], 
                                                        active_dims=[i for i in range(2, SPTVars.train_PT.shape[1])])) * \
                                        ScaleKernel(RQ(ard_num_dims=SPTVars.train_temp.shape[1], 
                                                        active_dims=[i for i in range(2, SPTVars.train_PT.shape[1])]))
                        phys_kernel =   ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                        ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) +\
                                        ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                        ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1]))
                        
                        # Initialize the STMTGP model
                        STMTGP = models.STMTGP(temporal_kernel=temp_kernel,
                                                full_kernel=full_kernel,
                                                physical_kernel=phys_kernel,
                                                CUDA=False, n_epochs=num_epochs,
                                                full_model_lr=GP_learning_rate)
                        
                        # Train STMTGP model
                        print("Training STMTGP model...")
                        STMTGP.train(SPTVars = SPTVars)

                        # Predict
                        STMTGP.predict(SPTVars = SPTVars)

                        # Reverse standardize the predictions
                        SPTVars.reverseStandardizeAlt(STMTGP)

                        STMTGP.fullMetricsAlt(SPTVars.full_pred_np, SPTVars.test_spat, SPTVars.test_spat_norm_tensor, lat_tc, verbose=verbose)
                    except:
                        print("Error in STMTGP model")
                        class STMTGP:
                            def __init__(self):
                                self.metrics = {
                                    'RMSE': np.nan,
                                    'MAE': np.nan,
                                    'MAD': np.nan,
                                    'MSLL': np.nan,
                                    'CL': np.nan,
                                    'DTW': np.nan
                                }
                                self.runtime_full = np.nan
                                self.runtime_full_pred = np.nan

                        STMTGP = STMTGP()
                
                # Sparse GP model
                if sparseGP: 
                    try:
                        print("Training Sparse GP model...")
                        SparseSTMTGP = models.SparseSTMTGP(n_epochs=num_epochs,
                                                            num_inducing=num_inducing,
                                                            num_latents=num_latents,
                                                            num_temp_inputs=SPTVars.train_temp.shape[1],
                                                            num_phys_inputs=SPTVars.train_phys.shape[1])
                        SparseSTMTGP.train(SPTVars = SPTVars)

                        # Predict
                        SparseSTMTGP.predict(SPTVars = SPTVars)

                        # Reverse standardize the predictions
                        SPTVars.reverseStandardizeAlt(SparseSTMTGP)

                        SparseSTMTGP.fullMetricsAlt(SPTVars.full_pred_np, SPTVars.test_spat, 
                                                    SPTVars.test_spat_norm_tensor, lat_tc, verbose=verbose)

                    except:
                        print("Error in Sparse GP model")
                        class SparseSTMTGP:
                            def __init__(self):
                                self.metrics = {
                                    'RMSE': np.nan,
                                    'MAE': np.nan,
                                    'MAD': np.nan,
                                    'MSLL': np.nan,
                                    'CL': np.nan,
                                    'DTW': np.nan
                                }
                                self.runtime_full = np.nan
                                self.runtime_full_pred = np.nan

                        SparseSTMTGP = SparseSTMTGP()
                
                if lstm: 
                    try:
                        # LSTM model
                        # Create DataLoader for training data
                        dataset = TensorDataset(SPTVars.train_temp, SPTVars.train_spat)
                        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                        # Initialize the model, loss function, and optimizer
                        model = BM.LSTMModel(input_size = SPTVars.train_temp.shape[1], 
                                            hidden_size = hidden_size, 
                                            num_layers = num_layers, 
                                            output_size = output_size)
                        loss_function = nn.MSELoss()
                        optimizer = optim.Adam(model.parameters(), lr=LSTM_learning_rate)

                        # Train the model
                        print("Training LSTM model...")
                        # Record time
                        lstm_start_time = tm.time()
                        model, loss_list = BM.LSTM_train(num_epochs, data_loader, model, loss_function, optimizer)

                        LSTM_mean, \
                            LSTM_lower, \
                            LSTM_upper, \
                                LSTM_std_dev = BM.MC_LSTM(model, 
                                                        SPTVars.test_temp, 
                                                        num_runs=num_lstm_runs)
                        lstm_end_time = tm.time()

                        # Reverse standardize the predictions
                        lstm_pred_np, \
                            lstm_lower_np, \
                            lstm_upper_np, \
                                lstm_std_dev_np = SPTVars.reverseStandardizeLSTM(LSTM_mean, 
                                                                                LSTM_lower, 
                                                                                LSTM_upper, 
                                                                                LSTM_std_dev)

                        # Get accuracy 
                        lstm_metrics = metrics.calculateMetricsAlt(lstm_pred_np, 
                                                                    lstm_std_dev_np, 
                                                                    SPTVars.test_spat, 
                                                                    lat_tc)

                    except:
                        print("Error in LSTM model")
                        lstm_metrics = {'RMSE': np.nan, 
                                        'MAE': np.nan, 
                                        'MAD': np.nan, 
                                        'MSLL': np.nan, 
                                        'CL': np.nan, 
                                        'DTW': np.nan}
                        lstm_end_time = lstm_start_time = 0

                # Add metrics to metrics_df 
                metrics_df = metrics_df.append({
                    'Fold': i,
                    'n_train': len(train),
                    'n_test': len(test),
                    'STMTGP_RMSE': STMTGP.metrics['RMSE'],
                    'STMTGP_MAE': STMTGP.metrics['MAE'],
                    'STMTGP_MAD': STMTGP.metrics['MAD'],
                    'STMTGP_MSLL': STMTGP.metrics['MSLL'],
                    'STMTGP_CL': STMTGP.metrics['CL'],
                    'STMTGP_DTW': STMTGP.metrics['DTW'],
                    'STMTGP_Train_Runtime': STMTGP.runtime_full,
                    'STMTGP_Pred_Runtime': STMTGP.runtime_full_pred,
                    'LSTM_RMSE': lstm_metrics['RMSE'],
                    'LSTM_MAE': lstm_metrics['MAE'],
                    'LSTM_MAD': lstm_metrics['MAD'],
                    'LSTM_MSLL': lstm_metrics['MSLL'],
                    'LSTM_CL': lstm_metrics['CL'],
                    'LSTM_DTW': lstm_metrics['DTW'],
                    'LSTM_Runtime': lstm_end_time - lstm_start_time,
                    'SparseSTMTGP_RMSE': SparseSTMTGP.metrics['RMSE'],
                    'SparseSTMTGP_MAE': SparseSTMTGP.metrics['MAE'],
                    'SparseSTMTGP_MAD': SparseSTMTGP.metrics['MAD'],
                    'SparseSTMTGP_MSLL': SparseSTMTGP.metrics['MSLL'],
                    'SparseSTMTGP_CL': SparseSTMTGP.metrics['CL'],
                    'SparseSTMTGP_DTW': SparseSTMTGP.metrics['DTW'],
                    'SparseSTMTGP_Train_Runtime': SparseSTMTGP.runtime_full,
                    'SparseSTMTGP_Pred_Runtime': SparseSTMTGP.runtime_full_pred,
                }, ignore_index=True)
            
            metrics_df.to_csv(csv_filename, index=False)

            # Also save to 'all_results' folder
            metrics_df.to_csv(user_path_results + csv_filename, index=False)
        print("Finished tests on User", id, "and Mode", label)
        print()

    print("Finished tests on all users and modes")  # End of script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_path_traj', type=str, help='Path to the input CSV file')
    parser.add_argument('--input_path_comp', type=str, help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--num_lstm_runs', type=int, default=70, help='Number of runs for the LSTM model')
    parser.add_argument('--GP_learning_rate', type=float, default=0.1, help='Learning rate for the GP model')
    parser.add_argument('--LSTM_learning_rate', type=float, default=0.01, help='Learning rate for the LSTM model')
    parser.add_argument('--num_inducing', type=int, default=200, help='Number of inducing points for the GP model')
    parser.add_argument('--compression_radius', type=float, default=0.2, help='Radius for compressing the trajectory data')
    parser.add_argument('--max_speed_filter', type=int, default=300, help='Maximum speed for filtering the data')
    parser.add_argument('--num_latents', type=int, default=3, help='Number of latents for the GP model')
    parser.add_argument('--n_MKL_epochs', type=int, default=20, help='Number of epochs for the MKL model')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth for the MKL tree')
    parser.add_argument('--hidden_size', type=int, default=48, help='Hidden size for the LSTM model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for the LSTM model')
    parser.add_argument('--output_size', type=int, default=2, help='Output size for the LSTM model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the LSTM model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print verbose output')

    args = parser.parse_args()

    main(args.input_path_traj, 
         args.input_path_comp,
         args.output_path, 
         args.num_epochs, 
         args.num_lstm_runs,
         args.GP_learning_rate,
         args.LSTM_learning_rate, 
         args.num_inducing, 
         args.compression_radius, 
         args.max_speed_filter, 
         args.num_latents, 
         args.n_MKL_epochs,
         args.max_depth,
         args.hidden_size, 
         args.num_layers, 
         args.output_size, 
         args.batch_size,
         args.seed,
         args.verbose)