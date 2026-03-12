import os
import pickle
import numpy as np

import bilby
from bilby.core.sampler.base_sampler import Sampler

try:
    import tempest as tp
except ImportError:
    tp = None


class Tempest(Sampler):

    sampler_name = "tempest"

    default_kwargs = dict(
        nlive=500,
        max_iters=None,
        checkpoint_interval=1000,
    )

    def __init__(self, likelihood, priors, outdir="outdir", label="label", **kwargs):

        super().__init__(likelihood, priors, outdir=outdir, label=label, **kwargs)

        if tp is None:
            raise ImportError("tempest-sampler must be installed")

        self.n_dim = len(self.priors)

        self.checkpoint_file = os.path.join(
            self.outdir,
            f"{self.label}_tempest_checkpoint.pkl",
        )

        self.resume = kwargs.get("resume", True)

        self._sampler = None

    # --------------------------------------------------
    # Bilby interface functions
    # --------------------------------------------------

    def prior_transform(self, u):

        params = []

        for i, key in enumerate(self.priors):
            params.append(self.priors[key].rescale(u[i]))

        return np.array(params)

    def log_likelihood(self, x):

        if x.ndim == 1:
            x = x[None, :]

        logl = []

        for row in x:
            params = dict(zip(self.priors.keys(), row))

            self.likelihood.parameters.update(params)

            logl.append(self.likelihood.log_likelihood())

        return np.array(logl)

    # --------------------------------------------------
    # Checkpoint helpers
    # --------------------------------------------------

    def save_checkpoint(self):

        if self._sampler is None:
            return

        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(self._sampler, f)

    def load_checkpoint(self):

        if not os.path.exists(self.checkpoint_file):
            return None

        with open(self.checkpoint_file, "rb") as f:
            sampler = pickle.load(f)

        return sampler

    # --------------------------------------------------

    def run_sampler(self):

        if self.resume:
            sampler = self.load_checkpoint()
        else:
            sampler = None

        if sampler is None:

            sampler = tp.Sampler(
                prior_transform=self.prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=self.n_dim,
                vectorize=True,
                nlive=self.kwargs.get("nlive"),
            )

        self._sampler = sampler

        max_iters = self.kwargs.get("max_iters")
        checkpoint_interval = self.kwargs.get("checkpoint_interval")

        iteration = 0

        while True:

            sampler.run(niter=checkpoint_interval)

            iteration += checkpoint_interval

            self.save_checkpoint()

            if max_iters is not None and iteration >= max_iters:
                break

            if sampler.is_finished():
                break

        samples, weights, logl = sampler.posterior()

        logz, logz_err = sampler.evidence()

        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err

        # Convert weighted posterior → equal weight
        weights = weights / np.sum(weights)

        idx = np.random.choice(
            np.arange(len(samples)),
            size=len(samples),
            p=weights,
        )

        self.result.samples = samples[idx]

        self.result.log_likelihood_evaluations = logl[idx]

        return self.result
