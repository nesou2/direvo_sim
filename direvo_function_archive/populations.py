"""
Contain the functions for generating, storing,
and mutating the populations.
The name is deceptive, this file is quite general 
and includes quite a lot of the logic for the evolutionary
algorithms. 
"""

import jax
import jax.numpy as jnp
import jax.random as jr


def gen_clustered_initial_population(rng, N, popsize, expected_diff, gene_options=2):
    """
    Generations an initial, clusteded population.
    Generates a point at random, then adds noise such that you
    have, on average, expected_diff mutations
    """
    r1, r2, r3 = jr.split(rng, 3)
    start_gene = jr.randint(r1, (N,), 0, gene_options)
    p = expected_diff/N
    mutations = jr.bernoulli(r2, p, (popsize, N))
    mutation_effects = jr.randint(r3, (popsize, N), 1, gene_options)
    return (start_gene + mutations*mutation_effects) % gene_options


def get_basic_pop_mutation(mutation_chance, num_options=2):
    """
    Applies a simple chance of mutating every gene in a population.
    """
    def to_ret(rng, pop):
        r1, r2, r3 = jr.split(rng, 3)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)
        mut_delta = jr.randint(r2, pshape, 1, num_options)
        return (pop + (has_mutation*mut_delta)) % num_options
    return to_ret



def alternating_strategy(strat1, strat2, first_len, second_len):
    """
    Gets the alternating strategy, which flips between strat1 and strat2
    """
    func1, state1 = strat1
    func2, state2 = strat2

    init_state = {"time left": first_len, "current strat" : 1 , "state 1" : state1, "state 2" : state2 }

    def decrement_strat_and_time(c_strat, time_left):
        new_strat = jax.lax.select(time_left == 1, 3 - c_strat, c_strat)
        new_time = jax.lax.select(time_left == 1, 
                        jax.lax.select(new_strat == 1, first_len, second_len),
                        time_left - 1 
                        )
        return new_strat, new_time

    def to_ret(rng, fitnesses, state):
        c_strat = state["current strat"]
        state_to_use = jax.lax.select( c_strat == 1, state["state 1"], state["state 2"] )

        fit_choice, new_inner_state = jax.lax.cond(c_strat == 1, func1, func2, rng, fitnesses, state_to_use)
        new_strat, new_time = decrement_strat_and_time(c_strat, state["time left"])

        new_s1 = jax.lax.select(c_strat == 1, new_inner_state, state["state 1"])
        new_s2 = jax.lax.select(c_strat == 1, state["state 2"],  new_inner_state)
        new_state = {"time left": new_time, "current strat" : new_strat , "state 1" : new_s1, "state 2" : new_s2 }
        return fit_choice, new_state
    
    return to_ret, init_state


def select_cells(selection_function, params):
    def to_ret(rng, fitnesses, state= 0):
        psize = fitnesses.shape[-1]
        selection_prob = selection_function(fitnesses, params)
        selected = jnp.ones((psize,))*jr.bernoulli(rng,
                                                   p=selection_prob, shape=(psize,))

        return jr.choice(rng, jnp.arange(psize), (psize,), p=selected), 0
    return to_ret, 0


def flow_select_cells(threshold_list):
    def to_ret(rng, fitnesses, state= 0):
        psize = fitnesses.shape[-1]
        assert fitnesses.shape[0] == len(threshold_list)
        num_lands = len(threshold_list)
        current_thresh = 0.0
        current_selected = jnp.ones((psize,), dtype=bool)
        for i, thresh in enumerate(threshold_list):
            current_fitness = jnp.where(current_selected, fitnesses[i], -jnp.inf)
            current_ranking = jnp.argsort(jnp.argsort(current_fitness)) / (current_fitness.shape[0]-1)
            
            current_thresh = 1.0 - (1.0 - current_thresh) * (1.0 - thresh)
            current_selected = current_selected * (current_ranking >= current_thresh)
        return jr.choice(rng, jnp.arange(psize), (psize,), p=current_selected), 0
    return to_ret, 0



def multi_select_wrapper(multi_fitnesses, params, state = 0):
    real_select = params['wrapped function']
    real_params = params['wrapped params']
    weighting = params['weighting']
    fitnesses = weighting @ multi_fitnesses
    res =  real_select(fitnesses, real_params)
    return res

def greedy_select(fitnesses, params):
    normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
    return 1.0* (normed_fitness >= params['threshold'])


# def spiking_select(fitnesses, params):
#     return params['base_chance'] + jnp.array(fitnesses >= jnp.percentile(fitnesses,
#                                                                  params['threshold']*100))*(1-params['base_chance'])
def spiking_select(fitnesses, params):
    normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
    return jnp.clip(params['base_chance'] + (normed_fitness >= params['threshold']), 0, 1)


def multi_get_fitness(single_fitness, xs, ds):
    return jnp.sum( (single_fitness >= xs) * ds)

vmap_get_fitness = jax.vmap(multi_get_fitness, in_axes= (0,None, None) )

def multi_spiking_select(fitnesses, params):
    psize = fitnesses.shape[0]
    rankings = jnp.argsort(jnp.argsort(fitnesses))/(psize-1)
    B = params["min chance"]
    F = params["max chance"]
    xs = params["step thresholds"]
    ds = params["step relative changes"]
    chances = B + (F - B) * vmap_get_fitness(rankings, xs, ds) / jnp.sum(ds)
    return chances


def sigmoid_select(fitnesses, params):
    normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
    return (1-params['base_chance']) / (1+jnp.exp(params['steepness'] *
                                ( params['threshold'] - normed_fitness))) + params['base_chance']


# Selection functions with no sub-sampling step.


def get_greedy_selection(top_k_to_take=1):
    """
    Gives a greedy top k selection function.
    """
    def to_ret(rng, fitnesses, state= 0):
        # Get the top k indicies, then repeat them enough times to fill out the population.
        psize = fitnesses.shape[0]
        indicies = jax.lax.top_k(fitnesses, top_k_to_take)[1]
        return jnp.repeat(indicies, psize//top_k_to_take + 1, total_repeat_length=psize), 0
    return to_ret, 0


def get_spiking_chance(greedy_frac, total_frac):
    """
    Takes the top greedy_frac, then fills 
    with other cells to reach total_frac.
    """
    spike_ratio = (total_frac - greedy_frac)*(1 - greedy_frac)

    def to_ret(fitnesses):
        normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
        best_members = normed_fitness >=  (1 - greedy_frac)
        return best_members * 1.0 + (1 - best_members) * spike_ratio
    return to_ret


def get_sigmoid_chance(param_array):
    """
    Gets a mutli-sigmoid chance.
    The param array is given by a list of (scale, steepness, centre )
    values. We normalize to the max being 1, so the best member of the poulation
    is always selected. 
    Also this acts on the normalized 0->1 percentile of fitnesses.
    """
    def to_ret(fitnesses):
        psize = fitnesses.shape[0]
        rankings = jnp.argsort(jnp.argsort(fitnesses))/(psize-1)
        total_masses = jnp.sum(jnp.array(
            [a[0] * jnp.arctan((rankings - a[2]) / a[1]) for a in param_array]), axis=0)
        return total_masses / jnp.max(total_masses)
    return to_ret


# rankings = jnp.argsort(jnp.argsort(fitnesses))/(psize-1)

def get_survival_selection(survival_chance_func):
    """
    Using survival chance func, kills cells, then
    regrows the population
    """
    def to_ret(rng, fitnesses, state= 0):
        r1, r2 = jr.split(rng)
        psize = fitnesses.shape[0]
        ps = survival_chance_func(fitnesses)
        survivors = jr.bernoulli(r1, ps)

        ordering = jnp.argsort(survivors)
        M = jnp.sum(survivors)
        # Used to ensure random cells make up the remaining bit.
        random_shift = jr.randint(r2, (1,), 0, M)[0]
        return ordering[psize - 1 - ((jnp.arange(psize) + random_shift) % M)], 0
    return to_ret, 0


def get_basic_softmax_selection(softmax_param=1.0, base_chance=0.0):
    """
    Gives a basic random softmax style selection
    """
    def to_ret(rng, fitnesses, state=0):
        # Takes the softmax of values as probabilities for choice
        # Then, adds on a base chance of selecting a uniform cell instead
        psize = fitnesses.shape[0]
        softmax_prob = jax.nn.softmax(softmax_param * fitnesses)
        final_prob = base_chance/psize + (1-base_chance)*softmax_prob
        return jr.choice(rng, jnp.arange(psize), (psize,),  p=final_prob), 0
    return to_ret, 0


def get_rank_softmax_selection(softmax_param=1.0, base_chance=0.0):
    """
    Gives a ranking-based random softmax style selection
    """
    def to_ret(rng, fitnesses, state= 0):
        # Takes the softmax of values as probabilities for choice
        # Then, adds on a base chance of selecting a uniform cell instead
        psize = fitnesses.shape[0]
        # Kinda weird, the argsort of the argsort is the ranking.
        rankings = jnp.argsort(jnp.argsort(fitnesses))/(psize-1)
        softmax_prob = jax.nn.softmax(softmax_param * rankings)
        final_prob = base_chance/psize + (1-base_chance)*softmax_prob
        return jr.choice(rng, jnp.arange(psize), (psize,),  p=final_prob), 0
    return to_ret, 0
