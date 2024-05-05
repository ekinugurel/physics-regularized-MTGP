import torch
import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RQKernel as RQ,  MaternKernel as MAT
from gpytorch.distributions import MultivariateNormal

num_latents = 3
num_tasks = 2
num_input_dims = 10
num_inducing = 100

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, temp=True, phys=False, likelihood=None,
                 num_tasks=num_tasks, num_latents=num_latents, num_input_dims=num_input_dims, num_inducing=num_inducing):
        
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, num_inducing, num_input_dims)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.likelihood = likelihood

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        if temp and not phys:
            self.covar_module = ScaleKernel(
                RQ(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            ) * ScaleKernel(
                RQ(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            )
        elif phys and not temp:
            self.covar_module = ScaleKernel(MAT(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
                                ScaleKernel(RQ(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
                                ScaleKernel(MAT(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
                                ScaleKernel(RQ(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents]))
        else:
            self.covar_module = ScaleKernel(
                RQ(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            ) * ScaleKernel(
                RQ(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            ) * \
            ScaleKernel(MAT(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
            ScaleKernel(RQ(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
            ScaleKernel(MAT(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents])) + \
            ScaleKernel(RQ(batch_shape=torch.Size([num_latents])), batch_shape=torch.Size([num_latents]))

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
class MKLSparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, likelihood=None,
                 num_tasks=num_tasks, num_latents=num_latents, num_input_dims=num_input_dims, num_inducing=num_inducing):
        
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, num_inducing, num_input_dims)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.likelihood = likelihood

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = kernel
        
    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)