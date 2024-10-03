import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from direvo_functions import *
from param_sampler import *
import selection_function_library as slct
import os
import tqdm

def directedEvolution(rng, 
                      selection_strategy, 
                      selection_params, 
                      empirical = False, 
                      N = None, 
                      K = None, 
                      landscape = None, 
                      popsize=100, 
                      mut_chance=0.01, 
                      num_steps=50, 
                      num_reps=10, 
                      define_i_pop=None, 
                      average=True):
    
    r1, r2 = jr.split(rng)
 
    # Get initial population.
    if define_i_pop == None:
        i_pop = np.array([jr.randint(r1, (N,), 0, 2)]*popsize)
    else:
        i_pop = define_i_pop
 
    # Function for evaluating fitness.
    if empirical:
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(r2, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)
 
    # Define selection function.
    selection_function = build_selection_function(
        selection_strategy, selection_params)
 
    # Bringing it all together.
    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function, fitness_function=fitness_function, num_steps=num_steps)[1]))
    
    # The array of seeds we will take as input.
    rng_seeds = jr.split(rng, num_reps)
    results = vmapped_run(rng_seeds)
 
    return results

thresholds, base_chances = base_chance_threshold_fixed_prop([0,0.19], 0.2, 5)
splits = [20,10,5,2,1]
muts = [0.01,0.1,0.5]
pops = [5000, 1000, 100]
Ns = [50,25]
K_ratio = [0.25, 0.2, 0.15, 0.1,0.05,0]

reps = 1

sweep_results = np.zeros((3,3,3,3,5,5))
total_iterations = len(pops) * len(muts) * len(Ns) * len(np.array(K_ratio) * max(Ns)) * len(splits) * len(base_chances)

with tqdm.tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for i, p in enumerate(pops):
        for ii, m in enumerate(muts):
            for iii, N in enumerate(Ns):
                for iv, K in enumerate(np.array(K_ratio)*N):
                    for v, s in enumerate(splits):
                        for vi, (bc, th) in enumerate(zip(base_chances, thresholds)):

                            repeat_results = []
                            rng_key = jr.PRNGKey(42)
                            rep_rngs = jr.split(rng_key, reps)

                            for rng in rep_rngs:
                                split_rngs = jr.split(rng, s)
                                split_results = []
                                for split in range(s):
                                    params = {'threshold': th, 'base_chance' : bc}
                                    run = directedEvolution(split_rngs[s], 
                                                            N = N, 
                                                            K=int(K), 
                                                            selection_strategy=slct.base_chance_threshold_select, 
                                                            selection_params = params, 
                                                            popsize=int(p/s), 
                                                            mut_chance=m/N, 
                                                            num_steps=100, 
                                                            num_reps=40, 
                                                            average=True)
                                
                                    #split_results.append(run['fitness'].max(axis=2).mean(axis=0)[-1])
                                    split_results.append(run['fitness'][:,:,-1].max(axis=1).mean())

                                repeat_results.append(np.array(split_results).max())

                            sweep_results[i,ii,iii,iv,v,vi] = np.mean(np.array(repeat_results))

                            # Update progress bar
                            pbar.update(1)

with open('strategy_sweep5.pkl', 'wb') as f:
    pickle.dump(sweep_results, f)