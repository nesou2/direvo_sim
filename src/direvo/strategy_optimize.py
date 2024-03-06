"""
Some basic algorithms for hyper-parameter optimization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from evosax import CMA_ES


def optimize_CMA_ES(rng, eval_func, num_params, normer = None, num_opt_steps = 20, ES_popsize = 20 ):
    eval_many = jax.jit(jax.vmap(eval_func, in_axes = (None, 0)))

    strategy = CMA_ES(popsize=ES_popsize, num_dims=num_params, elite_ratio=0.5)
    es_params = strategy.default_params
    
    if normer is None:
        normer = lambda x : x
        multi_normer = lambda x : x
    else:
        multi_normer = jax.jit(jax.vmap(normer))

    def evo_opt_to_scan(state, local_rng):
        rng_gen, rng_eval = jr.split(local_rng)
        x, state_1 = strategy.ask(rng_gen, state, es_params)

        fitness = -1.0*eval_many(rng_eval, multi_normer(x) ) 

        state_2 = strategy.tell(x, fitness, state_1, es_params)
        return (state_2, {"mean params" : normer(state_2.mean), "best fitness" : -1.0*state_2.best_fitness,
                          "current mean fitness" : -1.0*jnp.mean(fitness), "all fitnesses" : fitness, "parameters tested" : multi_normer(x) } )

    # We then get another state
    init_state = strategy.initialize(rng, es_params)

    results = jax.lax.scan(jax.jit(evo_opt_to_scan), init_state, jr.split(rng, num_opt_steps))

    extra_data = {"final optimization state" : results[0], "training history" : results[1]}
    best_param = normer(results[0].mean)

    return {"best param" : best_param , "extra info" : extra_data}


def optimize_grid_search(rng, eval_func, min_vals, max_vals, num_param_samples):
    eval_many = jax.jit(jax.vmap(eval_func, in_axes = (None, 0)))

    mesh_values = jnp.meshgrid(*[jnp.linspace(min_v, max_v, num_samps) for min_v, max_v, num_samps in zip(min_vals, max_vals, num_param_samples)])

    comb_arr = jnp.array([m.flatten() for m in mesh_values]).T

    evals = eval_many(rng,comb_arr) 

    best_arg = jnp.argmax(evals)

    extra_data = {"param array" : comb_arr, "performance array" : evals}
    return comb_arr[best_arg] , extra_data