from copy import deepcopy
from pathlib import Path
import inspect
import bilby
import numpy as np
from bilby.core.sampler.base_sampler import signal_wrapper
from bilby.core.utils.log import logger

try:
    import tempest as tp
except ImportError:
    tp = None

# Support bilby<2.7
try:
    from bilby.core.likelihood import _safe_likelihood_call
except ImportError:

    def _safe_likelihood_call(likelihood, params, use_ratio):
        """Fallback definition for bilby versions that do not have
        _safe_likelihood_call.
        """
        likelihood.parameters.update(params)
        return likelihood.log_likelihood()

def _log_likelihood_wrapper(theta):
    """Wrapper to the log likelihood that evaluates the prior constraints.

    Needed for multiprocessing."""
    from bilby.core.sampler.base_sampler import _sampling_convenience_dump

    theta = {
        key: theta[ii]
        for ii, key in enumerate(
            _sampling_convenience_dump.search_parameter_keys
        )
    }
    # bilby<2.7 compatibility
    try:
        params = deepcopy(_sampling_convenience_dump.parameters)
        params.update(theta)
    except AttributeError:
        params = theta

    if not _sampling_convenience_dump.priors.evaluate_constraints(theta):
        return -np.inf

    return _safe_likelihood_call(
        _sampling_convenience_dump.likelihood,
        params,
        _sampling_convenience_dump.use_ratio,
    )


class Tempest(bilby.core.sampler.Sampler):
    """Wrapper for tempest"""

    sampler_name = "tempest"

    sampling_seed_key = "random_state"

    @property
    def init_kwargs(self):
        params = inspect.signature(tp.Sampler).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        not_allowed = [
            "vectorize",
            "output_dir",
            "output_label",
            "n_dim",
            "pool",
            "reflective",  # Set automatically
            "periodic",  # Set automatically
        ]
        for key in not_allowed:
            kwargs.pop(key, None)
        return kwargs
      
    @property
    def run_kwargs(self):
        params = inspect.signature(tp.Sampler.run).parameters
        kwargs = {
            key: param.default
            for key, param in params.items()
            if param.default != param.empty
        }
        kwargs["save_every"] = 5
        return kwargs


    @property
    def default_kwargs(self):
        kwargs = self.init_kwargs
        kwargs.update(self.run_kwargs)
        kwargs["resume"] = True
        kwargs["npool"] = None
        return kwargs

    def _translate_kwargs(self, kwargs):
        """Translate the keyword arguments"""
        if "npool" not in kwargs:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["npool"] = kwargs.pop(equiv)
                    break
            # If nothing was found, set to npool but only if it is larger
            # than 1
            else:
                if self._npool > 1:
                    kwargs["npool"] = self._npool
        super()._translate_kwargs(kwargs)

    def _get_tempest_boundaries(self, key):
        # Based on the equivalent method for dynesty
        selected = list()
        for ii, param in enumerate(self.search_parameter_keys):
            if self.priors[param].boundary == key:
                logger.debug(f"Setting {key} boundary for {param}")
                selected.append(ii)
        if len(selected) == 0:
            selected = None
        return selected

    def prior_transform(self, u):
      return np.array([self.priors[k].rescale(u[i]) for i,k in enumerate(self.search_parameter_keys)])

    @signal_wrapper
    def run_sampler(self):

        init_kwargs = {k: self.kwargs.get(k) for k in self.init_kwargs.keys()}
        run_kwargs = {k: self.kwargs.get(k) for k in self.run_kwargs.keys()}
      
        output_dir = (
            Path(self.outdir) / f"{self.sampler_name}_{self.label}" / ""
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_pool()
        pool = self.kwargs.pop("pool", None)
        resume = self.kwargs.pop("resume", False)

        # Set the boundary conditions
        for key in ["reflective", "periodic"]:
            init_kwargs[key] = self._get_tempest_boundaries(key)

        if resume and run_kwargs["resume_state_path"] is None:
            resume_state_path = self._find_resume_state_path(output_dir)
            if resume_state_path is not None:
                logger.info(f"Resuming tempest from: {resume_state_path}")
                run_kwargs["resume_state_path"] = resume_state_path
            else:
                logger.debug("No files to resume from")

        sampler = tp.Sampler(
            prior_transform=self.prior_transform,
            log_likelihood=_log_likelihood_wrapper,
            output_label=self.label,
            output_dir=output_dir,
            n_dim=self.ndim,
            pool=self.pool,
            **init_kwargs,
        )

        sampler.run(**run_kwargs)
        self._close_pool()
      
        # Handle variable posterior() return signature
        posterior_output = sampler.posterior(resample=True)
        logger.info(
            f"posterior() returned {len(posterior_output)} items, "
            f"shapes: {[np.shape(r) for r in posterior_output]}"
        )
 
        if len(posterior_output) == 2:
            samples, logl = posterior_output
        elif len(posterior_output) == 3:
            samples, _, logl = posterior_output
 
        logz, logz_err = sampler.evidence()
        if logz_err is None:
            logz_err = np.nan
 
        logger.info(
            f"Tempest: log_evidence={logz:.3f} +/- {logz_err:.3f}, "
            f"n_samples={len(samples)}"
        )
 
        self.result.samples = samples
        self.result.log_likelihood_evaluations = logl
        self.result.log_evidence = logz
        self.result.log_evidence_err = logz_err
 
        return self.result

    def _find_resume_state_path(self, output_dir):
        files = list(output_dir.glob("*.state"))
        for file in files:
            if "final" in file.stem:
                logger.info("Found final state file")
                return file
        t_values = [int(file.stem.split("_")[-1]) for file in files]
        if len(t_values):
            t_max = max(t_values)
            state_path = output_dir / f"{self.label}_{t_max}.state"
            return state_path
        else:
            return None
   
