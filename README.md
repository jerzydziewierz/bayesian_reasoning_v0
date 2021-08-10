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
from bayesian_reasoning_v0.core import *
from lib.lib_nicefloat import *
```

```python
pd(observation_small_set_0)
```




    [[D=  0  ],
     [D=  0  ]]



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
    [[D=  0  ],
     [D=  0  ]]
    
    for parameter set      {φ}  = [φ=0.200  , φ=0.500  , φ=0.800  ]
    and prior belief set b({φ}) = [    333mR,     333mR,     333mR]
    
    for model parameter φ=0.200   with PriorBelief =     333mR, UpdatedBelief =     688mR, change of  +355mR 
    for model parameter φ=0.500   with PriorBelief =     333mR, UpdatedBelief =     269mR, change of   -64mR 
    for model parameter φ=0.800   with PriorBelief =     333mR, UpdatedBelief =      43mR, change of  -290mR 
    
    
                           {φ}  = [φ=0.200  , φ=0.500  , φ=0.800  ]
    posterior belief set b({φ}) = [    688mR,     269mR,      43mR]

