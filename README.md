# tempest-bilby

Bilby sampler plugin for the tempest nested sampler.

## Installation

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
