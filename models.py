"""
Author: Ekin Ugurel

Citation: 
"""

import gpytorch
import torch
import sgp
import GP
import time
import utils
from gpytorch.kernels import ScaleKernel, RQKernel as RQ, MaternKernel as MAT, PeriodicKernel as PER
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import numpy as np
import metrics

class STMTGP():
    def __init__(self, 
                 temporal_kernel = ScaleKernel(RQ(ard_num_dims=9, active_dims=[1,2,3,4,5,6,7,8,9])*PER(active_dims=[0])) \
                                    + ScaleKernel(RQ(ard_num_dims=9, active_dims = [1,2,3,4,5,6,7,8,9])*PER(active_dims=[0])),
                 full_kernel = ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                    ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) +\
                                    ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                    ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) * \
                                    ScaleKernel(RQ(ard_num_dims=9, active_dims=[3,4,5,6,7,8,9,10,11])*PER(active_dims=[2])) + \
                                    ScaleKernel(RQ(ard_num_dims=9, active_dims = [3,4,5,6,7,8,9,10,11])*PER(active_dims=[2])),
                 physical_kernel = ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                    ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])) +\
                                    ScaleKernel(MAT(nu=2.5, ard_num_dims=2, active_dims=[0,1])) + \
                                    ScaleKernel(RQ(ard_num_dims=2, active_dims=[0,1])),
                 likelihood=MultitaskGaussianLikelihood(num_tasks=2), 
                 n_epochs=50,
                 full_model_lr=0.1, 
                 CUDA=True,
                 equal_weights=False):
    
        self.comp_kernel_SP = self.comp_kernel_PS = physical_kernel
        self.comp_kernel_TS = temporal_kernel
        self.full_kernel = full_kernel
        self.likelihood = likelihood
        self.n_epochs = n_epochs    
        self.equal_weights = equal_weights
        self.full_model_lr = full_model_lr

        # If CUDA is available, use it
        if torch.cuda.is_available() & (CUDA == True):
            self.device = torch.device("cuda")
            self.likelihood = self.likelihood.cuda()
        else:
            self.device = torch.device("cpu")

    def __str__(self):
        return "STMTGP"
    
    def SPModel(self, 
                X_train, 
                y_train, 
                verbose=False, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Define spatial kernel and model
        """
        self.model_SP = GP.MTGPRegressor(X_train, y_train, self.comp_kernel_SP, num_tasks=2)
        if self.device == torch.device("cuda"):
            self.model_SP = self.model_SP.cuda()
        start_time = time.time()
        self.ls_SP, self.mll_SP = utils.training(self.model_SP, X_train, y_train, 
                                                 n_epochs=self.n_epochs, verbose=verbose, 
                                                 lr=lr, loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights)
        end_time = time.time()
        self.runtime_SP = end_time - start_time
        # Save model parameters
        self.model_SP_params = self.model_SP.state_dict()

    def SPModelPredict(self, X_test):
        """
        Predict speed/bearing with spatial model
        """
        self.model_SP.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad():
            self.spat_pred_dist = self.likelihood(self.model_SP(X_test))
            self.spat_mean = self.spat_pred_dist.mean
            self.spat_lower, self.spat_upper = self.spat_pred_dist.confidence_region()
        end=time.time()
        self.runtime_SP_pred = end - start
        if self.device == torch.device("cuda"):
            #self.spat_pred_dist = self.spat_pred_dist.cpu()
            self.spat_mean = self.spat_mean.cpu()
            self.spat_lower = self.spat_lower.cpu()
            self.spat_upper = self.spat_upper.cpu()

    def PSModel(self, 
                X_train, 
                y_train,
                verbose=False, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Define physical kernel and model
        """
        self.model_PS = GP.MTGPRegressor(X_train, y_train, self.comp_kernel_PS, num_tasks=2)
        if self.device == torch.device("cuda"):
            self.model_PS = self.model_PS.cuda()
        start_time = time.time()
        self.ls_PS, self.mll_PS = utils.training(self.model_PS, X_train, y_train, 
                                                 n_epochs=self.n_epochs, verbose=verbose, 
                                                 lr=lr, loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights)
        end_time = time.time()
        self.runtime_PS = end_time - start_time
        # Save model parameters
        self.model_PS_params = self.model_PS.state_dict()

    def PSModelPredict(self, X_test):
        """
        Predict lat/lng with physical model
        """
        self.model_PS.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad():
            self.phys_pred_dist = self.likelihood(self.model_PS(X_test))
            self.phys_mean = self.phys_pred_dist.mean
            self.phys_lower, self.phys_upper = self.phys_pred_dist.confidence_region()
        end=time.time()
        self.runtime_PS_pred = end - start
        if self.device == torch.device("cuda"):
            #self.phys_pred_dist = self.phys_pred_dist.cpu()
            self.phys_mean = self.phys_mean.cpu()
            self.phys_lower = self.phys_lower.cpu()
            self.phys_upper = self.phys_upper.cpu()

    def TSModel(self, 
                X_train, 
                y_train, 
                verbose=False,
                per_len_1 = 24*60*60,
                per_len_2 = 8*60*60, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Fit temporal kernel and model
        """
        # If kernel has a periodic component, set period length to 24 hours
        #if isinstance(self.comp_kernel_TS.kernels[1].kernels[0].base_kernel, PER):
        #    self.comp_kernel_TS.kernels[1].kernels[0].base_kernel.period_length = per_len_1
        #if isinstance(self.comp_kernel_TS.kernels[1].kernels[1].base_kernel, PER):
        #    self.comp_kernel_TS.kernels[1].kernels[1].base_kernel.period_length = per_len_2
        
        self.model_TS = GP.MTGPRegressor(X_train, y_train, self.comp_kernel_TS, num_tasks=2)
        if self.device == torch.device("cuda"):
            self.model_TS = self.model_TS.cuda()
        start_time = time.time()
        self.ls_TS, self.mll_TS = utils.training(self.model_TS, X_train, y_train, 
                                                 n_epochs=self.n_epochs, verbose=verbose, 
                                                 lr=lr, loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights)
        end_time = time.time()
        self.runtime_TS = end_time - start_time
        # Save model parameters
        self.model_TS_params = self.model_TS.state_dict()

    def TSModelPredict(self, X_test):
        """
        Predict lat/lng with temporal model
        """
        self.model_TS.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad():
            self.temp_pred_dist = self.likelihood(self.model_TS(X_test))
            self.temp_mean = self.temp_pred_dist.mean
            self.temp_lower, self.temp_upper = self.temp_pred_dist.confidence_region()
        end=time.time()
        self.runtime_TS_pred = end - start
        if self.device == torch.device("cuda"):
            #self.temp_pred_dist = self.temp_pred_dist.cpu()
            self.temp_mean = self.temp_mean.cpu()
            self.temp_lower = self.temp_lower.cpu()
            self.temp_upper = self.temp_upper.cpu()

    def SPTSFantasy(self):
        self.model_SP.eval()
        self.likelihood.eval()
        if self.device == torch.device("cuda"):
            self.temp_mean = self.temp_mean.cuda()
        with torch.no_grad():
            self.fantasy_pred_dist = self.likelihood(self.model_SP(self.temp_mean))
            self.fantasy_mean = self.fantasy_pred_dist.mean
            self.fantasy_lower, self.fantasy_upper = self.fantasy_pred_dist.confidence_region()
        #if self.device == torch.device("cuda"):
            #self.fantasy_pred_dist = self.fantasy_pred_dist.cpu()
            #self.fantasy_mean = self.fantasy_mean.cpu()
            #self.fantasy_lower = self.fantasy_lower.cpu()
            #self.fantasy_upper = self.fantasy_upper.cpu()

    def FullModel(self, 
                  X_train, 
                  y_train, 
                  per_len_1=24*60*60, 
                  per_len_2=8*60*60, 
                  verbose=False, 
                  loss_threshold=1e-4):
        """
        Fit full model
        """
        # TODO: Fix this
        #if isinstance(self.full_kernel.kernels[1].kernels[0].base_kernel, PER):
        #    self.full_kernel.kernels[1].kernels[0].base_kernel.period_length = per_len_1
        #if isinstance(self.full_kernel.kernels[1].kernels[1].base_kernel, PER):
        #    self.full_kernel.kernels[1].kernels[1].base_kernel.period_length = per_len_2
        
        self.full_model = GP.MTGPRegressor(X_train, y_train, self.full_kernel, num_tasks=2)
        if self.device == torch.device("cuda"):
            self.full_model = self.full_model.cuda()
        start_time = time.time()
        self.ls_full, self.mll_full = utils.training(self.full_model, X_train, y_train, 
                                                     n_epochs=self.n_epochs, 
                                                     verbose=verbose, lr=self.full_model_lr, 
                                                     loss_threshold=loss_threshold, 
                                                     equal_weights=self.equal_weights)
        end_time = time.time()
        self.runtime_full = end_time - start_time

    def FullModelPredict(self, X_test):
        """
        Predict lat/lng with full model
        """
        self.full_model.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad():
            self.full_pred_dist = self.likelihood(self.full_model(X_test))
            self.full_mean = self.full_pred_dist.mean
            self.full_lower, self.full_upper = self.full_pred_dist.confidence_region()
            self.std_dev = self.full_pred_dist.stddev
        
        end=time.time()
        self.runtime_full_pred = end - start
        if self.device == torch.device("cuda"):
            #self.full_pred_dist = self.full_pred_dist.cpu()
            self.full_mean = self.full_mean.cpu()
            self.full_lower = self.full_lower.cpu()
            self.full_upper = self.full_upper.cpu()
            self.std_dev = self.std_dev.cpu()

    def train(self, SPTVars):
        """
        Physics-Informed Multi-Task Gaussian Process for Human Mobility
        """
        # Fit phys-space model
        self.SPModel(SPTVars.train_spat, SPTVars.train_phys)

        # Fit time-space model
        self.TSModel(SPTVars.train_temp, SPTVars.train_spat)

        # Fit space-phys model
        self.PSModel(SPTVars.train_phys, SPTVars.train_spat)

        # Estimate lat/lng from posterior of time-space model
        self.TSModelPredict(SPTVars.test_temp)

        # Estimate speed/bearing from posterior of space-phys model
        self.PSModelPredict(SPTVars.test_phys)

        # Estimate speed/bearing from posterior of first model, using lat/lng predictions from second model
        self.SPTSFantasy()

        # Integrate speed/bearing predictions into original dataframe
        SPTVars.IntegrateFantasy(self.fantasy_mean)

        # Re-estimate lat/lng from using both spatial and temporal variables
        self.FullModel(SPTVars.train_PT, SPTVars.train_spat)

    def predict(self, SPTVars):
        """
        Predict lat/lng from speed/bearing predictions
        """
        self.FullModelPredict(SPTVars.test_fantasy)
        
    def tempMetrics(self, temp_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for temporal model
        """
        self.temp_rmse_lat, self.temp_rmse_lng, \
        self.temp_mae_lat, self.temp_mae_lng, \
        self.temp_nlpd, self.temp_msll_lat, self.temp_msll_lng, \
        self.temp_pcm, self.temp_df, self.temp_area, \
        self.temp_cl, self.temp_dtw  = metrics.calculateMetrics(
            self.temp_pred_dist, temp_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def physMetrics(self, phys_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for physical model
        """
        self.phys_rmse_lat, self.phys_rmse_lng, \
        self.phys_mae_lat, self.phys_mae_lng, \
        self.phys_nlpd, self.phys_msll_lat, self.phys_msll_lng, \
        self.phys_pcm, self.phys_df, self.phys_area, \
        self.phys_cl, self.phys_dtw  = metrics.calculateMetrics(
            self.phys_pred_dist, phys_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def fantasyMetrics(self, fantasy_preds_np, test_phys_np, test_phys_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for fantasy model
        """
        self.fantasy_rmse_lat, self.fantasy_rmse_lng, \
        self.fantasy_mae_lat, self.fantasy_mae_lng, \
        self.fantasy_nlpd, self.fantasy_msll_lat, self.fantasy_msll_lng, \
        self.fantasy_pcm, self.fantasy_df, self.fantasy_area, \
        self.fantasy_cl, self.fantasy_dtw = metrics.calculateMetrics(
            self.fantasy_pred_dist, fantasy_preds_np, test_phys_np, test_phys_norm_tensor, lat_tc, verbose=verbose)
        
    def fullMetrics(self, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for full model
        """
        self.full_rmse_lat, self.full_rmse_lng, \
        self.full_mae_lat, self.full_mae_lng, \
        self.full_nlpd, self.full_msll_lat, self.full_msll_lng, \
        self.full_pcm, self.full_df, self.full_area, \
        self.full_cl, self.full_dtw = metrics.calculateMetrics(
            self.full_pred_dist, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def fullMetricsAlt(self, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for full model
        """
        self.metrics = metrics.calculateMetricsAlt(
            full_preds_np, self.std_dev, test_spat_np, lat_tc, verbose=verbose)
        
class SPTVars():
    def __init__(self, train, test, scaler_speed, scaler_bearing, scaler_lat, scaler_lng, unix_col='unix_start_t', CUDA=True):
        self.train = train
        self.test = test
        self.scaler_speed = scaler_speed
        self.scaler_bearing = scaler_bearing
        self.scaler_lat = scaler_lat
        self.scaler_lng = scaler_lng

        self.train_spat = torch.tensor(train[['norm_lat', 'norm_lng']].values, dtype=torch.float32)
        self.test_spat = torch.tensor(test[['lat', 'lng']].values, dtype=torch.float32)
        self.test_spat_norm_tensor = torch.tensor(test[['norm_lat', 'norm_lng']].values, dtype=torch.float32)
        try:
            self.train_temp = torch.tensor(train[[unix_col, 'sam_sin', 'sam_cos', 'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6']].values, dtype=torch.float32)
            self.test_temp = torch.tensor(test[[unix_col, 'sam_sin', 'sam_cos', 'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6']].values, dtype=torch.float32)
        except:
            self.train_temp = torch.tensor(train[[unix_col, 'sam_sin', 'sam_cos']].values, dtype=torch.float32)
            self.test_temp = torch.tensor(test[[unix_col, 'sam_sin', 'sam_cos']].values, dtype=torch.float32)
            # Add the following columns
            self.train_temp = torch.cat((self.train_temp, torch.tensor(train[[col for col in train.columns if col.startswith('day_of_week')]].values, dtype=torch.float32)), dim=1)
            self.test_temp = torch.cat((self.test_temp, torch.tensor(test[[col for col in test.columns if col.startswith('day_of_week')]].values, dtype=torch.float32)), dim=1)
        self.train_phys = torch.tensor(train[['norm_speed', 'norm_bearing']].values, dtype=torch.float32)
        self.test_phys = torch.tensor(test[['vel', 'bearing']].values, dtype=torch.float32)
        self.test_phys_norm_tensor = torch.tensor(test[['norm_speed', 'norm_bearing']].values, dtype=torch.float32)
        
        self.train_ST = torch.cat((self.train_spat, self.train_temp), dim=1)
        self.test_ST = torch.cat((self.test_spat, self.test_temp), dim=1)
        self.train_PT = torch.cat((self.train_phys, self.train_temp), dim=1)
        self.test_PT = torch.cat((self.test_phys, self.test_temp), dim=1)

        # If CUDA is available, move tensors to GPU
        if torch.cuda.is_available() & CUDA:
            self.device = torch.device('cuda')
            self.train_spat = self.train_spat.cuda()
            self.test_spat = self.test_spat.cuda()
            self.train_temp = self.train_temp.cuda()
            self.test_temp = self.test_temp.cuda()
            self.train_phys = self.train_phys.cuda()
            self.test_phys = self.test_phys.cuda()
            self.train_ST = self.train_ST.cuda()
            self.test_ST = self.test_ST.cuda()
            self.train_PT = self.train_PT.cuda()
            self.test_PT = self.test_PT.cuda()
        else: 
            self.device = torch.device('cpu')

    def IntegrateFantasy(self, preds):
        #if torch.cuda.is_available():
            #self.test_temp = self.test_temp.cpu()

        # Cat preds to self.test_temp
        self.test_fantasy = torch.cat((preds, self.test_temp), dim=1)

        if self.device == torch.device('cuda'):
            self.test_fantasy = self.test_fantasy.cuda()

    def reverseStandardizePhys(self, preds, lower, upper):
        if self.device == torch.device('cuda'):
            preds = preds.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
        
        preds_speed = self.scaler_speed.inverse_transform(preds[:, 0].numpy().reshape(-1, 1))
        preds_bearing = self.scaler_bearing.inverse_transform(preds[:, 1].numpy().reshape(-1, 1))

        lower_speed = self.scaler_speed.inverse_transform(lower[:, 0].numpy().reshape(-1, 1))
        lower_bearing = self.scaler_bearing.inverse_transform(lower[:, 1].numpy().reshape(-1, 1))
        upper_speed = self.scaler_speed.inverse_transform(upper[:, 0].numpy().reshape(-1, 1))
        upper_bearing = self.scaler_bearing.inverse_transform(upper[:, 1].numpy().reshape(-1, 1))
        
        # Concatenate arrays
        pred_np = np.concatenate((preds_speed, preds_bearing), axis=1)
        lower = np.concatenate((lower_speed, lower_bearing), axis=1)
        upper = np.concatenate((upper_speed, upper_bearing), axis=1)

        return pred_np, lower, upper

    def reverseStandardizeSpat(self, preds, lower, upper):
        if self.device == torch.device('cuda'):
            preds = preds.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
       
        preds_lat = self.scaler_lat.inverse_transform(preds[:, 0].numpy().reshape(-1, 1))
        preds_lng = self.scaler_lng.inverse_transform(preds[:, 1].numpy().reshape(-1, 1))

        lower_lat = self.scaler_lat.inverse_transform(lower[:, 0].numpy().reshape(-1, 1))
        lower_lng = self.scaler_lng.inverse_transform(lower[:, 1].numpy().reshape(-1, 1))
        upper_lat = self.scaler_lat.inverse_transform(upper[:, 0].numpy().reshape(-1, 1))
        upper_lng = self.scaler_lng.inverse_transform(upper[:, 1].numpy().reshape(-1, 1))

        # Concatenate arrays
        pred_np = np.concatenate((preds_lat, preds_lng), axis=1)
        lower = np.concatenate((lower_lat, lower_lng), axis=1)
        upper = np.concatenate((upper_lat, upper_lng), axis=1)

        return pred_np, lower, upper
    
    def reverseStandardizeSpatAlt(self, preds, lower, upper, std_dev):
        if self.device == torch.device('cuda'):
            preds = preds.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
       
        preds_lat = self.scaler_lat.inverse_transform(preds[:, 0].numpy().reshape(-1, 1))
        preds_lng = self.scaler_lng.inverse_transform(preds[:, 1].numpy().reshape(-1, 1))

        lower_lat = self.scaler_lat.inverse_transform(lower[:, 0].numpy().reshape(-1, 1))
        lower_lng = self.scaler_lng.inverse_transform(lower[:, 1].numpy().reshape(-1, 1))
        upper_lat = self.scaler_lat.inverse_transform(upper[:, 0].numpy().reshape(-1, 1))
        upper_lng = self.scaler_lng.inverse_transform(upper[:, 1].numpy().reshape(-1, 1))

        std_dev_lat = self.scaler_lat.inverse_transform(std_dev[:, 0].numpy().reshape(-1, 1))
        std_dev_lng = self.scaler_lng.inverse_transform(std_dev[:, 1].numpy().reshape(-1, 1))

        # Concatenate arrays
        pred_np = np.concatenate((preds_lat, preds_lng), axis=1)
        lower = np.concatenate((lower_lat, lower_lng), axis=1)
        upper = np.concatenate((upper_lat, upper_lng), axis=1)
        std_dev = np.concatenate((std_dev_lat, std_dev_lng), axis=1)

        return pred_np, lower, upper, std_dev

    def reverseStandardize(self, STMTGP):
        self.temp_pred_np, self.temp_lower_np, self.temp_upper_np = \
            self.reverseStandardizeSpat(STMTGP.temp_mean, STMTGP.temp_lower, STMTGP.temp_upper)
        
        self.phys_pred_np, self.phys_lower_np, self.phys_upper_np = \
            self.reverseStandardizeSpat(STMTGP.phys_mean, STMTGP.phys_lower, STMTGP.phys_upper)

        self.fantasy_pred_np, self.fantasy_lower_np, self.fantasy_upper_np = \
            self.reverseStandardizePhys(STMTGP.fantasy_mean, STMTGP.fantasy_lower, STMTGP.fantasy_upper)
        
        self.full_pred_np, self.full_lower_np, self.full_upper_np = \
            self.reverseStandardizeSpat(STMTGP.full_mean, STMTGP.full_lower, STMTGP.full_upper)
        
    def reverseStandardizeAlt(self, STMTGP):
        self.full_pred_np, self.full_lower_np, self.full_upper_np, self.full_std_dev_np = \
            self.reverseStandardizeSpatAlt(STMTGP.full_mean, STMTGP.full_lower, STMTGP.full_upper, STMTGP.std_dev)
        
    def reverseStandardizeReg(self, mean, lower, upper, std_dev):
        self.full_pred_np, self.full_lower_np, self.full_upper_np, self.full_std_dev_np = \
            self.reverseStandardizeSpatAlt(mean, lower, upper, std_dev)
        
    def reverseStandardizeLSTM(self, preds, lower, upper, std_dev):
        if self.device == torch.device('cuda'):
            preds = preds.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
       
        preds_lat = self.scaler_lat.inverse_transform(preds[:, 0].reshape(-1, 1))
        preds_lng = self.scaler_lng.inverse_transform(preds[:, 1].reshape(-1, 1))

        lower_lat = self.scaler_lat.inverse_transform(lower[:, 0].reshape(-1, 1))
        lower_lng = self.scaler_lng.inverse_transform(lower[:, 1].reshape(-1, 1))
        upper_lat = self.scaler_lat.inverse_transform(upper[:, 0].reshape(-1, 1))
        upper_lng = self.scaler_lng.inverse_transform(upper[:, 1].reshape(-1, 1))

        std_dev_lat = self.scaler_lat.inverse_transform(std_dev[:, 0].reshape(-1, 1))
        std_dev_lng = self.scaler_lng.inverse_transform(std_dev[:, 1].reshape(-1, 1))

        # Concatenate arrays
        pred_np = np.concatenate((preds_lat, preds_lng), axis=1)
        lower = np.concatenate((lower_lat, lower_lng), axis=1)
        upper = np.concatenate((upper_lat, upper_lng), axis=1)
        std_dev = np.concatenate((std_dev_lat, std_dev_lng), axis=1)

        return pred_np, lower, upper, std_dev


class SparseSTMTGP():
    def __init__(self, 
                 likelihood=MultitaskGaussianLikelihood(num_tasks=2), 
                 num_latents=3,
                 num_temp_inputs=10,
                 num_phys_inputs=2,
                 num_inducing=100,
                 n_epochs=50, 
                 CUDA=True,
                 VariationalELBO = True,
                 equal_weights=False):

        self.likelihood = likelihood
        self.num_latents = num_latents
        self.num_temp_inputs = num_temp_inputs
        self.num_phys_inputs = num_phys_inputs
        self.num_inducing = num_inducing
        self.n_epochs = n_epochs    
        self.VariationalELBO = VariationalELBO
        self.equal_weights = equal_weights

        # If CUDA is available, use it
        if torch.cuda.is_available() & (CUDA == True):
            self.device = torch.device("cuda")
            self.likelihood = self.likelihood.cuda()
        else:
            self.device = torch.device("cpu")

    def __str__(self):
        return "STMTGP"
    
    def SPModel(self, 
                X_train, 
                y_train, 
                verbose=False, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Define spatial kernel and model
        """
        self.model_SP = sgp.MultitaskGPModel(temp=False, phys=True, likelihood=self.likelihood,
                                             num_tasks=2, num_latents=self.num_latents, 
                                             num_input_dims=self.num_phys_inputs, 
                                             num_inducing=self.num_inducing)
        if self.device == torch.device("cuda"):
            self.model_SP = self.model_SP.cuda()
        start_time = time.time()
        self.ls_SP, self.mll_SP = utils.training(self.model_SP, X_train, y_train, 
                                                 n_epochs=self.n_epochs, verbose=verbose, 
                                                 lr=lr, loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights,
                                                 VariationalELBO=self.VariationalELBO)
        end_time = time.time()
        self.runtime_SP = end_time - start_time
        # Save model parameters
        self.model_SP_params = self.model_SP.state_dict()

    def SPModelPredict(self, X_test):
        """
        Predict speed/bearing with spatial model
        """
        self.model_SP.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.spat_pred_dist = self.likelihood(self.model_SP(X_test))
            self.spat_mean = self.spat_pred_dist.mean
            self.spat_lower, self.spat_upper = self.spat_pred_dist.confidence_region()
        end=time.time()
        self.runtime_SP_pred = end - start
        if self.device == torch.device("cuda"):
            #self.spat_pred_dist = self.spat_pred_dist.cpu()
            self.spat_mean = self.spat_mean.cpu()
            self.spat_lower = self.spat_lower.cpu()
            self.spat_upper = self.spat_upper.cpu()

    def PSModel(self, 
                X_train, 
                y_train,
                verbose=False, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Define physical kernel and model
        """
        self.model_PS = sgp.MultitaskGPModel(temp=False, phys=True, likelihood=self.likelihood,
                                             num_tasks=2, num_latents=self.num_latents, 
                                             num_input_dims=self.num_phys_inputs, 
                                             num_inducing=self.num_inducing)
        if self.device == torch.device("cuda"):
            self.model_PS = self.model_PS.cuda()
        start_time = time.time()
        self.ls_PS, self.mll_PS = utils.training(self.model_PS, X_train, y_train, 
                                                 n_epochs=self.n_epochs, 
                                                 verbose=verbose, lr=lr, 
                                                 loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights,
                                                 VariationalELBO=self.VariationalELBO)
        end_time = time.time()
        self.runtime_PS = end_time - start_time
        # Save model parameters
        self.model_PS_params = self.model_PS.state_dict()

    def PSModelPredict(self, X_test):
        """
        Predict lat/lng with physical model
        """
        self.model_PS.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.phys_pred_dist = self.likelihood(self.model_PS(X_test))
            self.phys_mean = self.phys_pred_dist.mean
            self.phys_lower, self.phys_upper = self.phys_pred_dist.confidence_region()
        end=time.time()
        self.runtime_PS_pred = end - start
        if self.device == torch.device("cuda"):
            #self.phys_pred_dist = self.phys_pred_dist.cpu()
            self.phys_mean = self.phys_mean.cpu()
            self.phys_lower = self.phys_lower.cpu()
            self.phys_upper = self.phys_upper.cpu()

    def TSModel(self, 
                X_train, 
                y_train, 
                verbose=False,
                per_len_1 = 24*60*60,
                per_len_2 = 8*60*60, 
                lr=0.1, 
                loss_threshold=1e-4):
        """
        Fit temporal kernel and model
        """
        # If kernel has a periodic component, set period length to 24 hours
        #if isinstance(self.comp_kernel_TS.kernels[1].kernels[0].base_kernel, PER):
        #    self.comp_kernel_TS.kernels[1].kernels[0].base_kernel.period_length = per_len_1
        #if isinstance(self.comp_kernel_TS.kernels[1].kernels[1].base_kernel, PER):
        #    self.comp_kernel_TS.kernels[1].kernels[1].base_kernel.period_length = per_len_2
        
        self.model_TS = sgp.MultitaskGPModel(temp=True, phys=False, num_tasks=2,
                                             likelihood=self.likelihood,
                                             num_latents=self.num_latents, 
                                             num_input_dims=self.num_temp_inputs, 
                                             num_inducing=self.num_inducing)
        if self.device == torch.device("cuda"):
            self.model_TS = self.model_TS.cuda()
        start_time = time.time()
        self.ls_TS, self.mll_TS = utils.training(self.model_TS, X_train, y_train, 
                                                 n_epochs=self.n_epochs, 
                                                 verbose=verbose, lr=lr, 
                                                 loss_threshold=loss_threshold,
                                                 equal_weights=self.equal_weights,
                                                 VariationalELBO=self.VariationalELBO)
        end_time = time.time()
        self.runtime_TS = end_time - start_time
        # Save model parameters
        self.model_TS_params = self.model_TS.state_dict()

    def TSModelPredict(self, X_test):
        """
        Predict lat/lng with temporal model
        """
        self.model_TS.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.temp_pred_dist = self.likelihood(self.model_TS(X_test))
            self.temp_mean = self.temp_pred_dist.mean
            self.temp_lower, self.temp_upper = self.temp_pred_dist.confidence_region()
        end=time.time()
        self.runtime_TS_pred = end - start
        if self.device == torch.device("cuda"):
            #self.temp_pred_dist = self.temp_pred_dist.cpu()
            self.temp_mean = self.temp_mean.cpu()
            self.temp_lower = self.temp_lower.cpu()
            self.temp_upper = self.temp_upper.cpu()

    def SPTSFantasy(self):
        self.model_SP.eval()
        self.likelihood.eval()
        if self.device == torch.device("cuda"):
            self.temp_mean = self.temp_mean.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.fantasy_pred_dist = self.likelihood(self.model_SP(self.temp_mean))
            self.fantasy_mean = self.fantasy_pred_dist.mean
            self.fantasy_lower, self.fantasy_upper = self.fantasy_pred_dist.confidence_region()
        #if self.device == torch.device("cuda"):
            #self.fantasy_pred_dist = self.fantasy_pred_dist.cpu()
            #self.fantasy_mean = self.fantasy_mean.cpu()
            #self.fantasy_lower = self.fantasy_lower.cpu()
            #self.fantasy_upper = self.fantasy_upper.cpu()

    def FullModel(self, 
                  X_train, 
                  y_train, 
                  per_len_1=24*60*60, 
                  per_len_2=8*60*60, 
                  verbose=False, 
                  lr=0.1, 
                  loss_threshold=1e-4):
        """
        Fit full model
        """
        # TODO: Fix this
        #if isinstance(self.full_kernel.kernels[1].kernels[0].base_kernel, PER):
        #    self.full_kernel.kernels[1].kernels[0].base_kernel.period_length = per_len_1
        #if isinstance(self.full_kernel.kernels[1].kernels[1].base_kernel, PER):
        #    self.full_kernel.kernels[1].kernels[1].base_kernel.period_length = per_len_2
        
        self.full_model = sgp.MultitaskGPModel(temp=True, phys=True, num_tasks=2,
                                               likelihood=self.likelihood, 
                                                  num_latents=self.num_latents, 
                                                  num_input_dims=self.num_temp_inputs + self.num_phys_inputs, 
                                                  num_inducing=self.num_inducing)
        if self.device == torch.device("cuda"):
            self.full_model = self.full_model.cuda()
        start_time = time.time()
        self.ls_full, self.mll_full = utils.training(self.full_model, X_train, y_train, 
                                                     n_epochs=self.n_epochs, 
                                                     verbose=verbose, lr=lr, 
                                                     loss_threshold=loss_threshold, 
                                                     equal_weights=self.equal_weights,
                                                     VariationalELBO=self.VariationalELBO)
        end_time = time.time()
        self.runtime_full = end_time - start_time

    def FullModelPredict(self, X_test):
        """
        Predict lat/lng with full model
        """
        self.full_model.eval()
        self.likelihood.eval()
        start=time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.full_pred_dist = self.likelihood(self.full_model(X_test))
            self.full_mean = self.full_pred_dist.mean
            self.full_lower, self.full_upper = self.full_pred_dist.confidence_region()
            self.std_dev = self.full_pred_dist.stddev
        
        end=time.time()
        self.runtime_full_pred = end - start
        if self.device == torch.device("cuda"):
            #self.full_pred_dist = self.full_pred_dist.cpu()
            self.full_mean = self.full_mean.cpu()
            self.full_lower = self.full_lower.cpu()
            self.full_upper = self.full_upper.cpu()
            self.std_dev = self.std_dev.cpu()


    def train(self, SPTVars):
        """
        Physics-Informed Multi-Task Gaussian Process for Human Mobility
        """
        # Fit phys-space model
        self.SPModel(SPTVars.train_spat, SPTVars.train_phys)

        # Fit time-space model
        self.TSModel(SPTVars.train_temp, SPTVars.train_spat)

        # Fit space-phys model
        self.PSModel(SPTVars.train_phys, SPTVars.train_spat)

        # Estimate lat/lng from posterior of time-space model
        self.TSModelPredict(SPTVars.test_temp)

        # Estimate speed/bearing from posterior of space-phys model
        self.PSModelPredict(SPTVars.test_phys)

        # Estimate speed/bearing from posterior of first model, using lat/lng predictions from second model
        self.SPTSFantasy()

        # Integrate speed/bearing predictions into original dataframe
        SPTVars.IntegrateFantasy(self.fantasy_mean)

        # Re-estimate lat/lng from using both spatial and temporal variables
        self.FullModel(SPTVars.train_PT, SPTVars.train_spat)

    def predict(self, SPTVars):
        """
        Predict lat/lng from speed/bearing predictions
        """
        self.FullModelPredict(SPTVars.test_fantasy)
        
    def tempMetrics(self, temp_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for temporal model
        """
        self.temp_rmse_lat, self.temp_rmse_lng, \
        self.temp_mae_lat, self.temp_mae_lng, \
        self.temp_nlpd, self.temp_msll_lat, self.temp_msll_lng, \
        self.temp_pcm, self.temp_df, self.temp_area, \
        self.temp_cl, self.temp_dtw  = metrics.calculateMetricsAlt(
            self.temp_pred_dist, temp_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def physMetrics(self, phys_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for physical model
        """
        self.phys_rmse_lat, self.phys_rmse_lng, \
        self.phys_mae_lat, self.phys_mae_lng, \
        self.phys_nlpd, self.phys_msll_lat, self.phys_msll_lng, \
        self.phys_pcm, self.phys_df, self.phys_area, \
        self.phys_cl, self.phys_dtw  = metrics.calculateMetrics(
            self.phys_pred_dist, phys_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def fantasyMetrics(self, fantasy_preds_np, test_phys_np, test_phys_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for fantasy model
        """
        self.fantasy_rmse_lat, self.fantasy_rmse_lng, \
        self.fantasy_mae_lat, self.fantasy_mae_lng, \
        self.fantasy_nlpd, self.fantasy_msll_lat, self.fantasy_msll_lng, \
        self.fantasy_pcm, self.fantasy_df, self.fantasy_area, \
        self.fantasy_cl, self.fantasy_dtw = metrics.calculateMetrics(
            self.fantasy_pred_dist, fantasy_preds_np, test_phys_np, test_phys_norm_tensor, lat_tc, verbose=verbose)
        
    def fullMetrics(self, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for full model
        """
        self.full_rmse_lat, self.full_rmse_lng, \
        self.full_mae_lat, self.full_mae_lng, \
        self.full_nlpd, self.full_msll_lat, self.full_msll_lng, \
        self.full_pcm, self.full_df, self.full_area, \
        self.full_cl, self.full_dtw = metrics.calculateMetrics(
            self.full_pred_dist, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=verbose)
        
    def fullMetricsAlt(self, full_preds_np, test_spat_np, test_spat_norm_tensor, lat_tc, verbose=False):
        """
        Calculate metrics for full model
        """
        self.metrics = metrics.calculateMetricsAlt(
            full_preds_np, self.std_dev, test_spat_np, lat_tc, verbose=verbose)
        