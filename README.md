# Not a different way to do reasoning
> After reading this, you will be able to use Bayesian reasoning for your project.


## Who should read this

If you had a look at a pro toolbox like [PYMC3](https://docs.pymc.io/) and you feel like you banged yer head, this is the place to get it mended.

**My promise is this**: After reading this, you will be able to use Bayesian reasoning for your project, and use more advanced books. 

## What happened.

I do not normally write blog posts.

Today is not a normal day tough. 

I believe that I have found a missing link in my understanding of bayesian modelling theory.

This is not so much about the bayes belief update equation. It's good. The problem is that the notation is confusing, making it difficult to distinguish and source(compute, obtain) the various terms in the equation, and connect themto the real-world phenomena.

If you have been paying attention at school, you might think that bayes update rule:

{% raw %}
$$p( A | B) = \frac{p(B|A)p(A)}{p(B)}$$
{% endraw %}
> [wikipedia:Bayes Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)> > ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Bayes%27_Theorem_MMB_01.jpg/220px-Bayes%27_Theorem_MMB_01.jpg)

You might think that this equation is all that it takes.

By looking at the examples in the wiki, you might even get a feeling that you understand it. 

However, How does one apply this to problem X that is not described in the wiki?

It turns out that even professional data scientists (that I have met) are confused as to how to use this simple equation in practice.

In this write-up, I will explain why, what to do about it, and give a fairly universal live demo that can be adapted and extended for your problem. 




## Install the toolbox

`pip install git+https://github.com/jerzydziewierz/bayesian_reasoning_v0.git`

## How to use

Read the [chapter 1](https://jerzydziewierz.github.io/bayesian_reasoning_v0/10-gr-bayesian-modeling-missing-link.html).  (link here when I figure out how to add it)

Then use the the example provided to express your problem as a belief update problem. Profit!

## I want a quicker way.

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




    [??=0.200  , ??=0.500  , ??=0.800  ]



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
    
    for parameter set      {??}  = [??=0.200  , ??=0.500  , ??=0.800  ]
    and prior belief set b({??}) = [    333mR,     333mR,     333mR]
    
    for model parameter ??=0.200   with PriorBelief =     333mR, UpdatedBelief =     688mR, change of  +355mR 
    for model parameter ??=0.500   with PriorBelief =     333mR, UpdatedBelief =     269mR, change of   -64mR 
    for model parameter ??=0.800   with PriorBelief =     333mR, UpdatedBelief =      43mR, change of  -290mR 
    
    
                           {??}  = [??=0.200  , ??=0.500  , ??=0.800  ]
    posterior belief set b({??}) = [    688mR,     269mR,      43mR]


## I am not convinced

Contact me on LinkedIn and tell me why I am wrong.

https://www.linkedin.com/in/dr-george-rey-dziewierz/


---

