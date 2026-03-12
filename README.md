# tempest-bilby

Bilby sampler plugin for the tempest nested sampler.

## Installation

git clone https://github.com/mtagliazucchi/tempest-bilby
cd tempest-bilby
pip install -e .
## Usage

bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="tempest",
)

## Features

- Bilby-compatible sampler wrapper
- checkpointing
- resume support
- weighted posterior resampling
