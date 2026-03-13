import bilby
import numpy as np

rng = np.random.default_rng(1)

# Define a simple linear model with Gaussian likelihood and uniform priors
def model(x, m, c):
    return m * x + c

x = np.linspace(0, 10, 100)
injection_parameters = dict(m=0.5, c=0.2)
sigma = 1.0
y = model(x, **injection_parameters) + rng.normal(0.0, sigma, len(x))
likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)

priors = bilby.core.prior.PriorDict()
priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="tempest",
    n_total=5000,
    n_particles = 200,
    progress=True,
    vectorize=False
)
result.plot_corner()
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
