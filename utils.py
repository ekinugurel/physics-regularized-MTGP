"""
Author: Ekin Ugurel

Citation: 
"""

import torch
import matplotlib.pyplot as plt
import gpytorch
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import tqdm
import GP
import os
import sgp

def makeSeries(y_train_scaled, y_test_scaled, unix_min_tr, unix_min_te):
    lat = pd.Series(y_train_scaled[:,0].tolist(), unix_min_tr)
    lat_t = pd.Series(y_test_scaled[:,0].tolist(), unix_min_te)
    # Replace duplicates (in time) with the mean of the two values
    lat = lat.groupby(lat.index).mean().reset_index()
    lat = pd.Series(lat[0].tolist(), lat['index'].tolist())
    lat_tc = lat_t.groupby(lat_t.index).mean().reset_index() # For test set
    lat_tc = pd.Series(lat_tc[0].tolist(), lat_tc['index'].tolist())
    # Replace zeroes with positives close to zero
    lat.replace(0, 0.000000001, inplace=True)

    lon = pd.Series(y_train_scaled[:,1].tolist(), unix_min_tr)
    lon_t = pd.Series(y_test_scaled[:,1].tolist(),unix_min_te)
    # Replace duplicates (in time) with the mean of the two values
    lon = lon.groupby(lon.index).mean().reset_index()
    lon = pd.Series(lon[0].tolist(), lon['index'].tolist())
    lon_tc = lon_t.groupby(lon_t.index).mean().reset_index()
    lon_tc = pd.Series(lon_tc[0].tolist(), lon_tc['index'].tolist())
    # Replace zeroes with positives close to zero
    lon.replace(0, 0.000000001, inplace=True)
    return lat,lat_tc,lon,lon_tc

def tripLabelBasedTrainTestSplit(df, test_ratio=0.25, random_state=None):
    # Get unique trip identifiers (tids)
    unique_tids = df['tid'].unique()
    
    # Split the unique tids into training and testing sets
    train_tids, test_tids = train_test_split(
        unique_tids,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Filter the original dataframe based on the split tids
    train_df = df[df['tid'].isin(train_tids)]
    test_df = df[df['tid'].isin(test_tids)]
    
    return train_df, test_df

def tripLabelBasedKFoldSplit(df, k=5, random_state=None):
    # Get the unique trip IDs from the dataframe
    trip_ids = df['tid'].unique()
    
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # List to hold the training and testing dataframes for each fold
    k_folds = []
    
    # Perform the k-fold split on the trip IDs
    for train_idx, test_idx in kf.split(trip_ids):
        # Get the training and testing trip IDs based on the indices
        train_trip_ids = trip_ids[train_idx]
        test_trip_ids = trip_ids[test_idx]
        
        # Filter the original dataframe based on the training and testing trip IDs
        train_df = df[df['tid'].isin(train_trip_ids)]
        test_df = df[df['tid'].isin(test_trip_ids)]
        
        # Append the tuple of training and testing dataframes
        k_folds.append((train_df, test_df))
    
    return k_folds

def read_folder_names(input_path):
    # List to store the folder names
    folder_names = []

    # Loop through the data folder
    for folder_name in os.listdir(input_path + 'geolife/'):
        # Check if the folder name is a number
        if folder_name.isdigit():
            folder_names.append(int(folder_name))

    return folder_names

def process_data(points_m_c, data, points_m, 
                 upperbound=20, lowerbound=10,
                 random_state=42):
    cnt = 2
    if points_m_c['tid'].nunique() > upperbound:
        points_m_c_s = points_m_c[points_m_c['tid'].isin(points_m_c['tid'].sample(upperbound, random_state=random_state))]
        return process_data(points_m_c_s, data, points_m)
    elif points_m_c['tid'].nunique() < lowerbound:
        # Retain the top cnt clusters
        top_two = data['cluster'].value_counts().head(cnt).index
        top_two_data = data[data['cluster'].isin(top_two)]
        
        points_m_c = points_m[(points_m['latitudeStart'].isin(top_two_data['latitudeStart'])) & 
                              (points_m['longitudeStart'].isin(top_two_data['longitudeStart'])) & 
                              (points_m['latitudeEnd'].isin(top_two_data['latitudeEnd'])) & 
                              (points_m['longitudeEnd'].isin(top_two_data['longitudeEnd']))]
        cnt = cnt + 1
        return process_data(points_m_c, data, points_m)
    else:
        return points_m_c

def read_traj_data(input_path, id_user):
    # Create an empty dataframe to store the aggregated data
    aggregated_df = pd.DataFrame()

    if id_user < 100:
        folder_path = input_path + '/geolife/{}/Trajectory/'.format(id_user)
    else:
        folder_path = input_path + '/geolife/{}/Trajectory/'.format(id_user)

    # Loop through each .plt file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.plt'):
            # Read the .plt file into a dataframe
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delimiter=',', skiprows=6, header=None)

            # Drop the third column
            df = df.drop(columns=[2])

            # Remove .plt extension from the file name
            file_name = file_name.replace('.plt', '')

            # Set the name of the .plt file as the trip's unique identifier
            df['TripID'] = file_name

            # Make TripID numeric
            df['TripID'] = pd.to_numeric(df['TripID'], errors='coerce')

            # Rename the columns
            df.columns = ['lat', 'lon', 'alt', 'days', 'date', 'time', 'trip_id']

            # Append the dataframe to the aggregated dataframe
            aggregated_df = aggregated_df.append(df, ignore_index=True)
    return aggregated_df

def plot_kernel(kernel, xlim=None, ax=None):
    if xlim is None:
        xlim = [-3, 5]
    x = torch.linspace(xlim[0], xlim[1], 100)
    with torch.no_grad():
        K = kernel(x, torch.ones((1))).evaluate().reshape(-1, 1)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(x.numpy(), K.cpu().numpy())
    
def training(model, X_train, y_train, n_epochs=200, lr=0.1, 
             loss_threshold=0.00001, equal_weights = True, 
             fix_noise_variance=None, verbose=True,
             custom_per_lr=False, VariationalELBO=False): #sum_constraint=False):
    model.train()
    model.likelihood.train()
    
    if equal_weights:
        try:
            n_comp = len([m for m in model.covar_module.data_covar_module.kernels])
            for i in range(n_comp):
                model.covar_module.data_covar_module.kernels[i].outputscale = (1 / n_comp)
        except AttributeError:
            n_comp = 1

    # Use the adam optimizer
    if fix_noise_variance is not None:
        model.likelihood.noise = fix_noise_variance
        training_parameters = [p for name, p in model.named_parameters()
                               if not name.startswith('likelihood')]
    else:
        training_parameters = model.parameters()
        
    optimizer = torch.optim.Adam(training_parameters, lr=lr)

    if custom_per_lr:
        non_per_parameters = [p for name, p in model.named_parameters()
                               if not name.endswith("raw_period_length")]
        per1_parameters = [p for name, p in model.named_parameters()
                            if name.endswith("0.base_kernel.kernels.1.raw_period_length")]
        per2_parameters = [p for name, p in model.named_parameters()
                            if name.endswith("1.base_kernel.kernels.1.raw_period_length")]
        param_groups = [
            {'params': per1_parameters, 'lr': 3600},
            {'params': per2_parameters, 'lr': 1200},
            {'params': non_per_parameters, 'lr': 0.1},
            ]
        optimizer = torch.optim.Adam(param_groups, lr=lr)

    if VariationalELBO:
         # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=y_train.size(0))
    else:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    
    counter = 0
    ls = list()
    with tqdm.trange(n_epochs, disable=not verbose) as bar:
        for i in bar:
    
            optimizer.zero_grad()
            
            output = model(X_train)
            loss = -mll(output, y_train)
            if equal_weights:
                if hasattr(model.covar_module, 'data_covar_module'):
                    if hasattr(model.covar_module.data_covar_module, 'kernels'):
                        with torch.no_grad():
                            for j in range(n_comp):
                                model.covar_module.data_covar_module.kernels[j].outputscale =  \
                                model.covar_module.data_covar_module.kernels[j].outputscale /  \
                                sum([model.covar_module.data_covar_module.kernels[i].outputscale for i in range(n_comp)])
            #if sum_constraint:
            #    with torch.no_grad():
            #        for j in range(X_train.shape[0]):
            #            for k in range(4):
            #                output.mean[j][k] =  output.mean[j][k] / output.mean[j].sum(dim=0)
            #else:
            #    pass
            loss.backward()
            ls.append(loss.item())
            optimizer.step()
            if (i > 0):
                # If the loss decreased by less than the threshold for three iterations in a row, stop training.
                if (abs(ls[-1] - ls[-2]) < loss_threshold) and (abs(ls[-2] - ls[-3]) < loss_threshold):
                    break
            counter = counter + 1
                        
            # display progress bar
            postfix = dict(Loss=f"{loss.item():.3f}",
                           noise=f"{model.likelihood.noise.item():.3}")
            
            if (hasattr(model.covar_module, 'base_kernel') and
                hasattr(model.covar_module.base_kernel, 'lengthscale')):
                lengthscale = model.covar_module.base_kernel.lengthscale
                if lengthscale is not None:
                    lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
            else:
                lengthscale = model.covar_module.lengthscale

            #if lengthscale is not None:
            #    if len(lengthscale) > 1:
            #        lengthscale_repr = [f"{l:.3f}" for l in lengthscale]
            #        postfix['lengthscale'] = f"{lengthscale_repr}"
            #    else:
            #        postfix['lengthscale'] = f"{lengthscale[0]:.3f}"
                
            bar.set_postfix(postfix)
            
    return ls, mll
            
def train_model_get_bic(X_train, 
                        y_train, 
                        kernel, 
                        n_epochs=300, 
                        num_tasks=2, 
                        rank=1, 
                        lr=0.1, 
                        loss_threshold=1e-7, 
                        fix_noise_variance=None, 
                        sparse=False,
                        verbose=True):
    """
    Train GP model and calculate Bayesian Information Criterion (BIC)
    
    Parameters
    ----------
    model : gpytorch.models.ExactGP
        GP model

    X_train : torch.tensor
        Array of train features, n*d (d>=1)
    
    y_train : torch.tensor
        Array of target values
        
    kernel : gpytorch.kernels.Kernel
        Kernel object
        
    n_epochs : int
        Number of epochs to train GP model

    num_tasks : int
        Number of tasks

    rank : int
        Rank of the kernel
    
    lr : float
        Learning rate for Adam optimizer

    loss_threshold : float
        Threshold for stopping training

    fix_noise_variance : float
        Fix noise variance

    verbose : bool
        If True, display progress bar
        
    Returns
    -------
    bic : float
        BIC value
    ls: list
        List of losses during training
    """
    kernel = copy.deepcopy(kernel)
    if sparse:
        model = sgp.MKLSparseGPModel(kernel)
    else:
        model = GP.MTGPRegressor(X_train, y_train, kernel, num_tasks=num_tasks, rank=rank)

    try:
        n_comp = len([m for m in model.covar_module.data_covar_module.kernels])
        for i in range(n_comp):
            model.covar_module.data_covar_module.kernels[i].outputscale = (1 / n_comp)
    except AttributeError:
        n_comp = 1
    
    if sparse:
        ls, mll = training(model, X_train, y_train, n_epochs=n_epochs, verbose=verbose, lr=lr, loss_threshold=loss_threshold, fix_noise_variance=fix_noise_variance, VariationalELBO=True)
    else:
        ls, mll = training(model, X_train, y_train, n_epochs=n_epochs, verbose=verbose, lr=lr, loss_threshold=loss_threshold, fix_noise_variance=fix_noise_variance)
    
    with torch.no_grad():
        log_ll = mll(model(X_train), y_train) * X_train.shape[0]
        
    N = X_train.shape[0]
    m = sum(p.numel() for p in model.hyperparameters())
    bic = -2 * log_ll + m * np.log(N)

    return bic, ls 
    
def _get_all_product_kernels(op_list, kernel_list):
    """
    Find product pairs and calculate them.
    For example, if we are given expression:
        K = k1 * k2 + k3 * k4 * k5
    the function will calculate all the product kernels
        k_mul_1 = k1 * k2
        k_mul_2 = k3 * k4 * k5
    and return list [k_mul_1, k_mul_2].
    """
    product_index = np.where(np.array(op_list) == '*')[0]
    if len(product_index) == 0:
        return kernel_list

    product_index = product_index[0]
    product_kernel = kernel_list[product_index] * kernel_list[product_index + 1]
    
    if len(op_list) == product_index + 1:
        kernel_list_copy = kernel_list[:product_index] + [product_kernel]
        op_list_copy = op_list[:product_index]
    else:
        kernel_list_copy = kernel_list[:product_index] + [product_kernel] + kernel_list[product_index + 2:]
        op_list_copy = op_list[:product_index] + op_list[product_index + 1:]
        
    return _get_all_product_kernels(op_list_copy, kernel_list_copy)

def AdditiveKernelAlgebra(n_comp, start=0, kernel = gpytorch.kernels.Kernel):
    """
    Returns an additive kernel with n_comp components, each of which is a copy of kernel. Make the active dimensions of each component sequential by default.
    """
    kernel_list = []
    for i in range(start, start + n_comp):
        kernel_list.append(kernel(active_dims = torch.tensor([i])))
    return gpytorch.kernels.AdditiveKernel(*kernel_list)

def ConverttoTensor(X, y):
    """
    Convert numpy arrays to torch tensors
    """
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y