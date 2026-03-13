# tempest-bilby

A [Bilby](https://bilby-dev.github.io/bilby/) plugin that provides a sampler interface to [Tempest](https://tempest-sampler.readthedocs.io/en/latest/).

## Installation

Install both dependencies and then this package from source:

```bash
git clone https://github.com/mtagliazucchi/tempest-bilby.git
cd tempest-bilby
pip install -e .
```

## Quick start

### Simple Gaussian example

```python
import bilby
import numpy as np

rng = np.random.default_rng(1)

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
    n_particles=200,
    progress=True,
)
result.plot_corner()
```

### Gravitational wave / CBC example

See the ```tests``` folder!

## Sampler kwargs

All keyword arguments can be passed directly to `bilby.run_sampler`.

| Argument | Default | Description |
|---|---|---|
| `n_total` | `4096` | Total number of final samples |
| `n_particles` | `2 * n_dim` | Number of live particles |
| `vectorize` | `True` | If the likelihodd can handle input of shape `(n_particles, n_params)` |
| `progress` | `True` | Show tqdm progress bar |
| `resume_state_path` | `None` | Path to a checkpoint file to resume from |
| `save_every` | `None` | Save checkpoint every N iterations |
| `evaluate_constraints` | `True` | Enforce constrained prior support |

`pool` support will be soon be added!
