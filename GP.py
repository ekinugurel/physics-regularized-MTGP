"""
Author: Ekin Ugurel

Citation: 
"""

import torch
import gpytorch
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import MultitaskKernel
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


class MTGPRegressor(gpytorch.models.ExactGP):
    def __init__(self, X, y, kernel, mean=ConstantMean(), likelihood=None, num_tasks=2, rank=1):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        
        super().__init__(X, y, likelihood)
        self.mean = MultitaskMean(mean, num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(kernel, num_tasks=num_tasks, rank=rank)
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def predict(self, X):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self(X))
            mean = predictions.mean.numpy()
            self.lower, self.upper = predictions.confidence_region()
            self.lower = self.lower.numpy()
            self.upper = self.upper.numpy()
            std = predictions.stddev.numpy()
            self.sigma1_l = predictions.mean[:, 0] - predictions.stddev[:, 0]
            self.sigma1_u = predictions.mean[:, 0] + predictions.stddev[:, 0]
            self.sigma2_l = predictions.mean[:, 1] - predictions.stddev[:, 1]
            self.sigma2_u = predictions.mean[:, 1] + predictions.stddev[:, 1]

            # Convert to pandas dataframe
            mean_df = pd.DataFrame(mean)
            lower_df = pd.DataFrame(self.lower)
            upper_df = pd.DataFrame(self.upper)
            std_df = pd.DataFrame(std)

            #mean_df['sum'] = mean_df.sum(axis=1)
        return mean_df, lower_df, upper_df, std_df
    
    def plot_preds(self, mean, date_train, date_test, y_train, y_test, 
                   label1 = 'training data', label2 = 'predictions', figsize = (10, 5)):
        plot_df = pd.DataFrame({'mean_lat': mean[:,0],
                        'mean_long': mean[:,1],
                        'lower_lat': self.lower[:,0],
                        'lower_long': self.lower[:,1],
                        'upper_lat': self.upper[:,0],
                        'upper_long': self.upper[:,1],
                        'sigma1_l': self.sigma1_l,
                        'sigma1_u': self.sigma1_u,
                        'sigma2_l': self.sigma2_l,
                        'sigma2_u': self.sigma2_u,
                       'datetime': date_test},
                       columns=['mean_lat', 'mean_long', 'lower_lat', 
                                'lower_long', 'upper_lat', 'upper_long', 
                                'sigma1_l', 'sigma1_u', 'sigma2_l', 'sigma2_u', 'datetime'])

        # Initialize plots
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        f, (y1_ax, y2_ax) = plt.subplots(2, 1, constrained_layout = True)

        y1_ax.plot(date_train, y_train[:,0].numpy(), '.', c = 'blue', label = label1)
        y1_ax.plot(date_test, plot_df['mean_lat'], '.', c='red', label = label2)
        y1_ax.scatter(date_test, y_test[:,0].numpy(), marker='.', c='blue')
        y1_ax.fill_between(date_test, 0, 1, where=date_test, 
                           color='pink', alpha=0.5, label = 'Testing period', 
                           transform=y1_ax.get_xaxis_transform())
        y1_ax.set_title('Latitude')
        y1_ax.set_xticks([])

        y2_ax.plot(date_train, y_train[:,1].numpy(), '.', c = 'blue', label = label1)
        y2_ax.plot(date_test, plot_df['mean_long'], '.', c='red', label = label2)
        y2_ax.scatter(date_test, y_test[:,1].numpy(), marker='.', c='blue')
        y2_ax.fill_between(date_test, 0, 1, where=date_test, 
                           color='pink', alpha=0.5, label = 'Testing period', 
                           transform=y2_ax.get_xaxis_transform())
        y2_ax.set_title('Longitude')
        #y2_ax.set_xticks([])

        plt.legend()
        plt.xticks(rotation = 45)
        plt.xlabel('Date', fontsize=10)

        plt.show()