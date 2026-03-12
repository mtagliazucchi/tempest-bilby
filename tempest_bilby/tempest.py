import numpy as np
import pandas as pd
import bilby
from bilby.core.sampler.base_sampler import Sampler

try:
    import tempest as tp
except ImportError:
    tp = None


class Tempest(Sampler):

    sampler_name = "tempest"

    # arguments users can pass through bilby.run_sampler()
    default_kwargs = dict(
        n_total=4096,
        n_particles = None,
        progress=True,
        resume_state_path=None,
        save_every=None,
        vectorize=False,
    )

    def __init__(self, likelihood, priors, **kwargs):

        super().__init__(likelihood, priors, **kwargs)

        if tp is None:
            raise ImportError("tempest-sampler must be installed")

        self.n_dim = len(self.priors)

        self.kwargs["n_particles"] = self.kwargs.get("n_particles", 2 * self.n_dim)
    # -------------------------
    # Prior transform
    # -------------------------

    def prior_transform(self, u):

        params = []

        for i, key in enumerate(self.priors):
            params.append(self.priors[key].rescale(u[i]))

        return np.array(params)

    # -------------------------
    # Log-likelihood
    # -------------------------

    def log_likelihood(self, x):

        if x.ndim == 1:
            x = x[None, :]

        logl = []

        for row in x:

            params = dict(zip(self.priors.keys(), row))

            self.likelihood.parameters.update(params)

            logl.append(self.likelihood.log_likelihood())

        return np.array(logl)

    # -------------------------
    # Run sampler
    # -------------------------

    def run_sampler(self):

        # Tempest constructor arguments
        sampler = tp.Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=self.log_likelihood,
            n_dim=self.n_dim,
            n_particles = self.kwargs["n_particles"],
            vectorize=self.kwargs["vectorize"],
        )

        # Tempest run() arguments
        run_kwargs = {}
    
        for key in ["n_total", "progress", "resume_state_path", "save_every"]:
            value = self.kwargs.get(key)
            if value is not None:
                run_kwargs[key] = value
    
        sampler.run(**run_kwargs)
      
        samples, _, logl = sampler.posterior(resample=True)
        logz, logz_err = sampler.evidence()

        if logz_err is None:
            logz_err = np.nan

        # Include the log likelihood and log prior in the samples
        # so that we can populate the result object correctly
        samples = pd.DataFrame(samples, columns=self.search_parameter_keys)
        samples["log_likelihood"] = logl
    
        self.result.samples = posterior_samples.drop(
            columns=["log_likelihood"]
        ).values
        self.result.log_likelihood_evaluations = posterior_samples[
            "log_likelihood"
        ].values
        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err
        
        return self.result
