import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from selection_function_library import *
from direvo_functions import *
from param_sampler import *

def selection_function_plotter(function, params, label, color):
    
    fitness = jnp.linspace(0,1,num=1000)
    selection_chance = function(fitness,params)

    plt.plot(fitness, selection_chance, label = label, color = color)
    plt.ylabel('Selection chance', fontsize=8)
    plt.xlabel('Fitness percentile', fontsize=8)
    plt.title(f'Selection function', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.ylim(0,1.1)

def quatromatrix(left, bottom, right, top, ax=None, tripcolorkw={}):
    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()
    
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, cmap='viridis', **tripcolorkw)
    return tripcolor

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