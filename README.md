# Project name here
> Summary description here.


```python
%load_ext autoreload
%autoreload 2
```

This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

```python
1+1
```




    2



```python
import bayesian_reasoning_v0.core
dir(bayesian_reasoning_v0.core)
```




    ['BayesianBeliefUpdate',
     'BayesianBeliefUpdateReport',
     'BayesianBeliefUpdate_singlePhi',
     '__all__',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'coin_model',
     'functools',
     'ja',
     'jax',
     'lib_nicefloat',
     'likelihood_observation_set_fn',
     'likelihood_vector_fn',
     'marginal_fn',
     'nicefloat',
     'numpy',
     'observation_large_set_biased',
     'observation_large_set_neutral',
     'observation_single_0',
     'observation_single_1',
     'observation_small_set_0',
     'observation_small_set_1',
     'observation_small_set_balanced',
     'pb',
     'pd',
     'pe',
     'pf',
     'phi_set',
     'pl',
     'prior_belief_set_bias_0',
     'prior_belief_set_bias_1',
     'prior_belief_set_fair',
     'prior_belief_set_uninformative']



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

    for observations {D}: 
    [[D=  0  , ],
     [D=  0  , ]]
    
    for parameter set      {ϕ}  = [ϕ=0.200  , ϕ=0.500  , ϕ=0.800  , ]
    and prior belief set b({ϕ}) = [    333mR,     333mR,     333mR, ]
    
    for model parameter ϕ=0.200  ,  with PriorBelief =     333mR, , UpdatedBelief =     688mR, , change of  +355mR 
    for model parameter ϕ=0.500  ,  with PriorBelief =     333mR, , UpdatedBelief =     269mR, , change of   -64mR 
    for model parameter ϕ=0.800  ,  with PriorBelief =     333mR, , UpdatedBelief =      43mR, , change of  -290mR 
    
    
                           {ϕ}  = [ϕ=0.200  , ϕ=0.500  , ϕ=0.800  , ]
    posterior belief set b({ϕ}) = [    688mR,     269mR,      43mR, ]

