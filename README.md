# Intro to Bayesian reasoning
> After reading this, you will be able to use Bayesian reasoning for your project.


```python
%load_ext autoreload
%autoreload 2
```

This file will become your README and also the index of your documentation.

## Install

`pip install git+https://github.com/jerzydziewierz/bayesian_reasoning_v0.git`

## How to use

```python
import bayesian_reasoning_v0
dir(bayesian_reasoning_v0)
```

```python
this_observation_set = observation_small_set_0
this_phi_prior_belief_set = prior_belief_set_uninformative

posterior_belief_set=BayesianBeliefUpdate(
                        phi_set = phi_set,
                        phi_prior_belief_set=this_phi_prior_belief_set,
                        observations=this_observation_set,
                        likelihood_fn=likelihood_observation_set_fn,
                        marginal_probability_fn = marginal_fn)

BayesianBeliefUpdateReport(
                        phi_set = phi_set,
                        phi_prior_belief_set=this_phi_prior_belief_set,
                        observations=this_observation_set,
                        posterior_belief_set=posterior_belief_set)

```
