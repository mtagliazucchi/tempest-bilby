import bilby
import numpy as np

class GaussianLikelihood(bilby.Likelihood):

    def __init__(self):
        parameters = dict(x=0)
        super().__init__(parameters)

    def log_likelihood(self):
        x = self.parameters["x"]
        return -0.5 * x**2


priors = dict(
    x=bilby.core.prior.Uniform(-5,5)
)

likelihood = GaussianLikelihood()

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="tempest",
    nlive=200,
)
result.plot_corner()
