import numpy as np
import bilby
from bilby.core.sampler.base_sampler import Sampler, signal_wrapper

try:
    import tempest as tp
except ImportError:
    tp = None


def _make_prior_transform(priors, search_parameter_keys):
    """
    Return a closure that maps a unit-cube vector u -> physical parameter vector.
    Uses bilby prior .rescale() (= ppf) for each parameter.
    Does NOT apply any conversion_function — that belongs in the likelihood wrapper.
    """
    def prior_transform(u):
        result = np.empty(len(search_parameter_keys))
        for i, key in enumerate(search_parameter_keys):
            result[i] = priors[key].rescale(u[i])
        return result
    return prior_transform


def _make_log_likelihood(likelihood, search_parameter_keys):
    """
    Return a closure compatible with both tempest calling conventions:

    - vectorize=True  → called once with x of shape (n_particles, n_dim),
                        must return array of shape (n_particles,)
    - vectorize=False → called via map() with x of shape (n_dim,),
                        must return a scalar float

    The conversion_function (if any) is attached to the likelihood object
    before this closure is created.
    """
    conversion_function = getattr(likelihood, "conversion_function", None)

    def _eval_single(xi):
        """Evaluate log-likelihood for one parameter vector xi (1-D)."""
        params = {k: float(xi[j]) for j, k in enumerate(search_parameter_keys)}
        if conversion_function is not None:
            params, _ = conversion_function(params)
        likelihood.parameters.update(params)
        return likelihood.log_likelihood()

    def log_likelihood(x):
        x = np.atleast_1d(x)
        if x.ndim == 1:
            # Single particle call (vectorize=False, called via map)
            return _eval_single(x)
        else:
            # Batched call (vectorize=True)
            return np.array([_eval_single(xi) for xi in x])

    return log_likelihood


class Tempest(Sampler):
    """Bilby wrapper for the Tempest importance-nested sampler."""

    sampler_name = "tempest"

    # All kwargs accepted by tp.Sampler.__init__ and tp.Sampler.run
    default_kwargs = dict(
        n_total=4096,
        n_particles=None,
        progress=True,
        resume_state_path=None,
        save_every=None,
        vectorize=True,       # tempest is designed for vectorised calls
        evaluate_constraints=True,
    )

    # kwargs that belong to tp.Sampler.__init__ (not .run())
    _init_kwargs = {"n_particles", "vectorize"}
    # kwargs that belong to .run()
    _run_kwargs = {"n_total", "progress", "resume_state_path", "save_every"}

    def __init__(self, likelihood, priors, outdir="outdir", label="label",
                 use_ratio=False, plot=False, skip_import_verification=False,
                 conversion_function=None, **kwargs):
        if tp is None:
            raise ImportError(
                "tempest-sampler must be installed: pip install tempest-sampler"
            )

        super().__init__(
            likelihood, priors,
            outdir=outdir, label=label,
            use_ratio=use_ratio, plot=plot,
            skip_import_verification=skip_import_verification,
            **kwargs,
        )

        # Attach conversion function to the likelihood so the logl closure can use it
        if conversion_function is not None:
            self.likelihood.conversion_function = conversion_function

        # Default n_particles to 2 * ndim if not provided
        if self.kwargs.get("n_particles") is None:
            self.kwargs["n_particles"] = max(2 * self.ndim, 10)

    def prior_transform(self, u):
        """Transform unit-cube sample u -> physical parameter array."""
        return _make_prior_transform(self.priors, self.search_parameter_keys)(u)

    @signal_wrapper
    def run_sampler(self):
        # Build clean closures — no global state, no private bilby internals
        prior_transform_fn = _make_prior_transform(
            self.priors, self.search_parameter_keys
        )
        log_likelihood_fn = _make_log_likelihood(
            self.likelihood, self.search_parameter_keys
        )

        # Split kwargs into init and run groups
        init_kwargs = {k: self.kwargs[k] for k in self._init_kwargs
                       if k in self.kwargs and self.kwargs[k] is not None}
        run_kwargs = {k: self.kwargs[k] for k in self._run_kwargs
                      if k in self.kwargs and self.kwargs[k] is not None}

        bilby.core.utils.logger.info(
            f"Starting Tempest sampler with n_dim={self.ndim}, "
            f"n_particles={init_kwargs.get('n_particles')}, "
            f"n_total={run_kwargs.get('n_total')}"
        )

        # Initialise Tempest
        sampler = tp.Sampler(
            prior_transform=prior_transform_fn,
            log_likelihood=log_likelihood_fn,
            n_dim=self.ndim,
            **init_kwargs,
        )

        # Run
        sampler.run(**run_kwargs)

        # Extract results
        samples, _, logl = sampler.posterior(resample=True)
        logz, logz_err = sampler.evidence()
        if logz_err is None:
            logz_err = np.nan

        bilby.core.utils.logger.info(
            f"Tempest finished: log_evidence={logz:.3f} +/- {logz_err:.3f}, "
            f"n_samples={len(samples)}"
        )

        # Populate bilby result
        self.result.samples = samples
        self.result.log_likelihood_evaluations = logl
        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err

        return self.result
