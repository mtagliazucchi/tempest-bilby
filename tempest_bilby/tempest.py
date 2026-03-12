import numpy as np
import bilby
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.sampler.base_sampler import signal_wrapper

try:
    import tempest as tp
except ImportError:
    tp = None


def prior_transform(u, priors, search_parameter_keys, conversion_function=None):
    params = {}
    for i, key in enumerate(search_parameter_keys):
        prior = priors[key]
        if hasattr(prior, "ppf"):
            params[key] = prior.ppf(u[i])
        else:
            # fallback for composite priors
            sample = priors.sample_subset_constrained(keys=[key], size=1)
            if isinstance(sample, list):
                sample = sample[0]
            params[key] = sample[key]

    if conversion_function is not None:
        converted_params = conversion_function(params)
        params.update(converted_params)

    return np.array([params[k] for k in search_parameter_keys])

def _tempest_log_likelihood(theta):
    """Wrap likelihood to respect bilby search_parameter_keys and conversion_function."""
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    # Map vector -> dict using search_parameter_keys
    params = {k: theta[i] for i, k in enumerate(_sampling_convenience_dump.search_parameter_keys)}

    # Apply Bilby conversion function if it exists (e.g., compute mass_1, mass_2 from mc, eta)
    conv_fn = getattr(_sampling_convenience_dump.likelihood, "conversion_function", None)
    if conv_fn is not None:
        params = conv_fn(params)

    # Update likelihood and compute log-likelihood
    _sampling_convenience_dump.likelihood.parameters.update(params)
    return _sampling_convenience_dump.likelihood.log_likelihood()
  
class Tempest(Sampler):
    """Bilby wrapper for Tempest sampler."""
    sampler_name = "tempest"

    default_kwargs = dict(
        n_total=4096,
        n_particles=None,
        progress=True,
        resume_state_path=None,
        save_every=None,
        vectorize=False,
        evaluate_constraints=True,
    )

    def __init__(self, likelihood, priors, **kwargs):
        if tp is None:
            raise ImportError("tempest-sampler must be installed")

        # Call base Sampler init
        super().__init__(likelihood, priors, **kwargs)

        # Attach the conversion_function if it exists in kwargs
        conv_fn = kwargs.get("conversion_function", None)
        if conv_fn is not None:
            self.likelihood.conversion_function = conv_fn

        # Number of particles
        self.kwargs["n_particles"] = self.kwargs.get("n_particles", 2 * self.ndim)
    
    def prior_transform(self, u):
        return prior_transform(
            u,
            self.priors,
            self.search_parameter_keys,
            conversion_function=self.conversion_function
        )
  
    @signal_wrapper
    def run_sampler(self):
        from bilby.core.sampler.base_sampler import _sampling_convenience_dump

        # Make likelihood globally accessible for _tempest_log_likelihood
        _sampling_convenience_dump.likelihood = self.likelihood
        _sampling_convenience_dump.search_parameter_keys = self.search_parameter_keys

        sampler = tp.Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=_tempest_log_likelihood,
            n_dim=self.ndim,
            n_particles=self.kwargs["n_particles"],
            vectorize=self.kwargs.get("vectorize", False),
        )

        run_kwargs = {k: v for k, v in self.kwargs.items()
                      if k in ["n_total", "progress", "resume_state_path", "save_every"] and v is not None}

        sampler.run(**run_kwargs)

        # Extract posterior and evidence
        samples, _, logl = sampler.posterior(resample=True)
        logz, logz_err = sampler.evidence()
        if logz_err is None:
            logz_err = np.nan

        self.result.samples = samples
        self.result.log_likelihood_evaluations = logl
        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err
        return self.result
