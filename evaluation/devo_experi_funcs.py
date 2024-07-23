from . import directedevo as devo

import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr

import pandas as pd

import itertools

def nk_simple_runs(rng, N,K, popsize, softmax_p, mut_param, n_runs, r_len):
    def keep_prob_func(x):
        return jax.nn.softmax(x*softmax_p)

    mut_prob = (- jnp.log(mut_param))/N

    def to_vmap(r):
        r1, r2, r3 = jr.split(r, 3)
        start_point = devo.gen_clustered_inital_population(r1, N, popsize, 1)
        my_landscape = devo.gen_NK_func(r2, N, K)
        return devo.do_dir_evo_run(r3, start_point, my_landscape, keep_prob_func, mut_prob, r_len)[1]

    multiple_runs = jax.vmap(to_vmap)
    return np.array(multiple_runs(jr.split(rng, n_runs)))

def nk_simple_runs_2(rng, N,K, popsize, softmax_p, mut_param, n_runs, r_len):
    def keep_prob_func(x):
        return jax.nn.softmax(x*softmax_p)

    mut_prob = (- jnp.log(mut_param))/N

    def to_vmap(r):
        r1, r2, r3 = jr.split(r, 3)
        start_point = devo.gen_clustered_inital_population(r1, N, popsize, 1)
        my_landscape = devo.gen_NK_func(r2, N, K)
        return devo.do_dir_evo_run(r3, start_point, my_landscape, keep_prob_func, mut_prob, r_len)[1]

    multiple_runs = jax.vmap(to_vmap)
    return multiple_runs(jr.split(rng, n_runs))

def get_nk_fitness_func(N, K, popsize, num_reps, run_length):
    """
    Gets the fitness functions,
    which takes the rng, softmax param, and
    mutation param.
    """
    def fitty_func(rng, input_params):
        softmax_para, mut_param = input_params[0], input_params[1]
        def keep_prob_func(x):
            return jax.nn.softmax(x*softmax_para)

        mut_prob = (- jnp.log(mut_param))/N

        def to_vmap(r):
            r1, r2, r3 = jr.split(r, 3)
            start_point = devo.gen_clustered_inital_population(r1, N, popsize, 1)
            my_landscape = devo.gen_NK_func(r2, N, K)
            return devo.do_dir_evo_run(r3, start_point, my_landscape, keep_prob_func, mut_prob, run_length)[1]

        multiple_runs = jax.vmap(to_vmap)
        return jnp.mean(multiple_runs(jr.split(rng, num_reps)))

    #Poor naming, we need to vmap across the inputs 
    return jax.jit(jax.vmap(fitty_func, in_axes=(None, 0)))

def get_nk_fitness_func2(N, K, popsize, num_reps, run_length):
    """
    Gets the fitness functions,
    which takes the rng, softmax param, and
    mutation param.
    Adds another param, a base keep chance.
    """
    def fitty_func(rng, input_params):
        relued_input = jax.nn.relu(input_params)
        softmax_para, mut_param, base_chance = relued_input[0], relued_input[1], relued_input[2]
        def keep_prob_func(x):
            return base_chance + (1 - base_chance)*jax.nn.softmax(x*softmax_para)

        mut_prob = (- jnp.log(mut_param))/N

        def to_vmap(r):
            r1, r2, r3 = jr.split(r, 3)
            start_point = devo.gen_clustered_inital_population(r1, N, popsize, 1)
            my_landscape = devo.gen_NK_func(r2, N, K)
            return devo.do_dir_evo_run(r3, start_point, my_landscape, keep_prob_func, mut_prob, run_length)[1]

        multiple_runs = jax.vmap(to_vmap)
        return jnp.mean(multiple_runs(jr.split(rng, num_reps)))

    #Poor naming, we need to vmap across the inputs 
    return jax.jit(jax.vmap(fitty_func, in_axes=(None, 0)))


def run_simple_experi(runs = 100, r_len = 100):
    n_runs_for_each = runs
    run_length = r_len

    N_vals = [20,20,40,100]
    k_vals = [3,10,20,30]

    nk_pairs = list(zip(N_vals, k_vals))

    pop_sizes = [50, 100, 200]


    softmax_params = [1e1, 1e0, 1e-1, 1e-2]

    # chance of mutiation pass changing nothing
    mut_params = [10e-2, 7e-2, 3e-2, 1e-2 ]

    params_iterator = itertools.product(nk_pairs, pop_sizes, softmax_params, mut_params)

    curves_all = []
    NK_all = []
    pop_all = []
    soft_all = []
    mut_all = []

    rng = jr.PRNGKey(0)

    for nk, pop, soft_p, mut_p in params_iterator:
        NK_all.append(nk)
        pop_all.append(pop)
        soft_all.append(soft_p)
        mut_all.append(mut_p)
        curves_all.append(nk_simple_runs(rng, nk[0], nk[1], pop, soft_p, mut_p, n_runs_for_each, run_length))

    dicty = {"NK" : NK_all, "population" : pop_all, "softmax param": soft_all , "mutation rate": mut_all , "results" : curves_all}
    return dicty
    









