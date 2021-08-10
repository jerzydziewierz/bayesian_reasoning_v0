# What happened.



> After reading this, you will be able to use Bayesian reasoning for your project.

## Install

`pip install git+https://github.com/jerzydziewierz/bayesian_reasoning_v0.git`

## How to use

Read the [chapter 1](10-gr-bayesian-modeling-missing-link.ipynb).  (link here when I figure out how to add it)

Then use the the example provided to express your problem as a belief update problem. Profit!

# I want a quicker way.

Here you go:

```python
from bayesian_reasoning_v0.core import *
from lib.lib_nicefloat import *
import numpy

ja=lambda x: jax.numpy.array(x)
```

Assume that our plant is a classic coin. 

Assume that we can model it with a hidden parameter $\phi$

```python
def coin_model(real_hidden_phi=0.5 ,observation_count=10):
    """
    a model of a coin: simulates coin tosses.
    
    """
    observations = jax.numpy.array(numpy.random.random(observation_count)<real_hidden_phi,dtype=jax.numpy.int32).reshape(observation_count,1)
    return observations

pd(coin_model(observation_count = 3))
```




    [[D=  1  ],
     [D=  1  ],
     [D=  1  ]]



In this example, we will consider 3 possible values for $\phi$ : $\{\phi\} = [0.2, 0.5, 0.8] $

```python
# # # Hidden parameter - suspected values to evaluate
# note that the convention is to put phis in column headers column headers
phi_set = ja([0.2,0.5,0.8])
pf(phi_set)
```




    [φ=0.200  , φ=0.500  , φ=0.800  ]



Here are possible prior beliefs to try:

```python
# like with phi, we put prior and posterior beliefs in column headers (columns, or 1D-only arrays)

# no bias belief: we think that the coin is not biased.
prior_belief_set_fair = ja([0.1,0.8,0.1])

# high bias belief: we think that the coin is biased.
prior_belief_set_bias_0 = ja([0.8,0.1,0.1])

# high bias belief: we think that the coin is biased towards ones.
prior_belief_set_bias_1 = jax.numpy.array([0.1,0.1,0.8])

# uninformative: We know that we don't know. 
prior_belief_set_uninformative = jax.numpy.array([0.3333,0.3333,0.3333])
```

```python
pb(prior_belief_set_uninformative)
```




    [    333mR,     333mR,     333mR]



For repeatability, here are example observation sets to try:

```python
# # # Observations  -- constants or generate using coin_model
# note that the convention is to put observations in rows of the table.
observation_single_1 = ja([[1]]) # note: need a categorical variable here - and only "1" or "0"
observation_single_0 = ja([[0]]) # note: need a categorical variable here - and only "1" or "0"
observation_small_set_0 = ja([[0],[0]])
observation_small_set_balanced = ja([[0],[1]])
observation_small_set_1 = ja([[1],[1]])
observation_large_set_neutral = ja([[1],[1],[1],[1],[0],[0],[0],[0]])
observation_large_set_biased = ja([[1],[1],[1],[1],[0],[0]])
```

```python
pd(observation_small_set_balanced)
```




    [[D=  0  ],
     [D=  1  ]]



```python
# now, place the observations and prior beliefs here, and enjoy the results.

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


# I am not convinced

Contact me on LinkedIn and tell me why I am wrong.

https://www.linkedin.com/in/dr-george-rey-dziewierz/

