"""
Inlcudes functions for evaluating strategies on a given environment.
All of this code is designed such that it possible to exactly specify
the evolutionary experiment you are running. 
This is probably excessive for most use cases! 
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

import chex
import typing
import pickle
from chex import PRNGKey

from . import populations as pop
from . import landscapes as ld
from . import evo_runs


# The new env param class.
# Designed to be more flexible, and hence allow for more
# complex things later down the line. 
@chex.dataclass
class EnvParams:
    mutation_generator: "typing.Callable[[EnvParams], typing.Callable]"
    mutation_params: typing.Any
    landscape_generator: "typing.Callable[[PRNGKey, EnvParams], typing.Callable]"
    landscape_params: typing.Any
    initial_pop_generator: "typing.Callable[[PRNGKey, EnvParams, typing.Callable], typing.Callable]"
    initial_pop_params: typing.Any
    extra_info: typing.Any = None
    # gene_size: int
    # popsize: int
    
    
@chex.dataclass
class BasicMutationParams:
    mutation_chance: float
    num_options: int
    num_elites : int = 0
    
@chex.dataclass
class NKLandscapeParams:
    gene_size: int
    K : int
    popsize: int
    distribution: typing.Callable
    multi_land: bool = False
    num_lands : int = 1

@chex.dataclass
class FixedLandParams:
    landscape: chex.Array
    gene_size: int
    popsize: int
    use_codon: bool = False
    shuffle_landscape : bool = False
    shuffle_fraction : float = 0.0
    mask_number:int = 0
    multi_land: bool = False

@chex.dataclass
class BasicStartingLocationParams:
    initial_spread: float = 0.0 
    fixed_start: bool = False
    starting_location: chex.Array = None
    
@chex.dataclass
class PreOptimizeStartingLocationParams:
    number_of_steps: int = 10
    inner_popsize: int = 100
    initial_spread: float = 0.0 
    fixed_start: bool = False
    starting_location: chex.Array = None
    multi_land: bool = False
    multi_weighting: chex.Array = None


def basic_mutation_generator(env_params: EnvParams):
    return pop.get_basic_pop_mutation(env_params.mutation_params.mutation_chance, env_params.mutation_params.num_options)
    
#mut_method = pop.get_basic_pop_mutation(land_params.mutation_chance, num_options)
def nk_land_generator(rng, env_params : EnvParams):
    """
    Generates a NK landscape
    """
    if env_params.landscape_params.multi_land:
        rs = jr.split(rng, env_params.landscape_params.num_lands)
        functions = [ld.gen_NK_func(r, env_params.landscape_params.gene_size, env_params.landscape_params.K, env_params.landscape_params.distribution) for r in rs]
        return function_map(functions)
    else:
        return ld.gen_NK_func(rng, env_params.landscape_params.gene_size, env_params.landscape_params.K, env_params.landscape_params.distribution)
    
def function_map(func_list):
    def to_ret(*args, **kwargs):
        return jnp.array([func(*args, **kwargs) for func in func_list])
    return to_ret

def pre_opt_init_pop_generator(rng, env_params : EnvParams, landscape_func):
    """
    Gens a starting location for the population
    which has already been optimized greedily
    """
    
    mut_method = pop.get_basic_pop_mutation(0.5/env_params.landscape_params.gene_size, env_params.mutation_params.num_options)
    r1, r2 = jr.split(rng)
    if env_params.initial_pop_params.fixed_start:
        i_pop = fixed_init_pop_generator(r1, env_params, landscape_func)
    else:
        i_pop = basic_init_pop_generator(r1, env_params, landscape_func)
    if env_params.initial_pop_params.multi_land:
        weighting = env_params.initial_pop_params.multi_weighting
        my_select_params = {"wrapped function" : pop.greedy_select, "wrapped params" : {"threshold" : 0.95}, "weighting" : weighting}
        selection_pair = pop.select_cells(pop.multi_select_wrapper, my_select_params)
    else:
        selection_pair = pop.get_greedy_selection(1)
    result = evo_runs.cont_evo_run(r2, i_pop, selection_pair[1], selection_pair[0], mut_method, landscape_func, env_params.initial_pop_params.number_of_steps)
    return result[0][0]

def fixed_init_pop_generator(rng, env_params : EnvParams, landscape_func):
    """
    Generates a fixed starting location for the population
    """
    init_pop = env_params.initial_pop_params.starting_location
    popsize = env_params.landscape_params.popsize
    starting_location = init_pop * jnp.ones( (popsize, env_params.landscape_params.gene_size), dtype=int)
    return starting_location

def basic_init_pop_generator(rng, env_params : EnvParams, landscape_func):
    """
    Generates a starting location for the population
    with some limited spread
    """
    return pop.gen_clustered_initial_population(rng, env_params.landscape_params.gene_size, env_params.landscape_params.popsize, env_params.initial_pop_params.initial_spread, env_params.mutation_params.num_options)

#@partial(jax.jit, static_argnums=(2,))
def randomly_shuffle_land(rng, a, fraction_to_shuffle, to_fix):
    N = np.prod(np.array(a.shape))
    to_fix_flat = jnp.ravel_multi_index(to_fix, a.shape, mode = "wrap")
    num_to_shuffle = np.int32(N * fraction_to_shuffle)
    to_shuffle = jr.choice(rng, N - 1, shape=(num_to_shuffle,), replace=False)
    to_shuffle = (to_shuffle + 1 + to_fix_flat) % N
    # Applys a cyclic shift
    to_shuffle_to = jnp.roll(to_shuffle, 1)
    
    ind1 = jnp.unravel_index(to_shuffle, a.shape)
    ind2 = jnp.unravel_index(to_shuffle_to, a.shape)
    
    return a.at[ind1].set(a[ind2])

def fixed_land_generator(rng, env_params : EnvParams):
    """
    Generates a fixed landscape
    """
    if env_params.landscape_params.multi_land:
        all_lands = env_params.landscape_params.landscape
        num_of_lands = all_lands.shape[0]
        rngs = jr.split(rng, num_of_lands)
        land_func_list = [get_landscape_function(r, all_lands[i], env_params) for i, r in enumerate(rngs)]
        return function_map(land_func_list) 
    else:
        return get_landscape_function(rng, env_params.landscape_params.landscape, env_params)


def get_landscape_function(rng, input_landscape, env_params : EnvParams):
    if env_params.initial_pop_params.fixed_start:
        starting_loc = env_params.initial_pop_params.starting_location
    else:
        num_dims = env_params.landscape_params.gene_size
        num_options = env_params.mutation_params.num_options
        r_start, rng = jr.split(rng)
        starting_loc = jr.randint(r_start, (num_dims,), minval= np.zeros((num_dims,)), maxval=np.array(num_options) )
    if env_params.landscape_params.use_codon:
        stuff = starting_loc.reshape(-1,3)
        post_trans_start = ld.CODON_MAPPER[stuff[:,0],stuff[:,1],stuff[:,2]]
    else:
        post_trans_start = starting_loc
        
    if env_params.landscape_params.shuffle_landscape:
        r_shuffle, rng = jr.split(rng)
        landscape = randomly_shuffle_land(r_shuffle, input_landscape, env_params.landscape_params.shuffle_fraction, post_trans_start)
    else:
        landscape = input_landscape
    
    mask_number = env_params.landscape_params.mask_number
    if mask_number > -1:
        num_dims = env_params.landscape_params.gene_size
        r_mask, rng = jr.split(rng)
        mask = jnp.zeros((num_dims,), dtype=bool)
        mask = mask.at[:mask_number].set(True)
        mask = jr.permutation(r_mask, mask)
        if env_params.landscape_params.use_codon:
            return ld.get_pd_landscape_function_codon_masked(landscape, mask, starting_loc)
        else:
            return ld.get_pd_landscape_function_masked(landscape, mask, starting_loc)
    else:
        if env_params.landscape_params.use_codon:
            return ld.get_pre_defined_landscape_function_with_codon(landscape)
        else:
            return ld.get_pre_defined_landscape_function(landscape)
    
def get_basic_NK_params(N:int, 
                        K:int,
                        popsize : int, 
                        mutation_rate_normed : float, 
                        dist = jr.normal,
                        greedy_init_steps = 0, 
                        num_lands = 1, 
                        land_weighting = None, 
                        num_gene_options = 2,
                        num_elites :int = 0,
                        ) -> EnvParams:
    mutation_rate = mutation_rate_normed/N
    mut_params = BasicMutationParams(mutation_chance = mutation_rate, num_options = num_gene_options, num_elites= num_elites)
    doing_multi_land = num_lands > 1
    if doing_multi_land and land_weighting is None:
        land_weighting = jnp.ones((num_lands,))
    land_params = NKLandscapeParams(gene_size = N, K=K, popsize = popsize, distribution= dist, num_lands = num_lands, multi_land=doing_multi_land)
    if greedy_init_steps > 0:
        init_pop_func = pre_opt_init_pop_generator
        init_pop_params = PreOptimizeStartingLocationParams(number_of_steps = greedy_init_steps, inner_popsize = 100, initial_spread = 0.1, multi_land=doing_multi_land, multi_weighting = land_weighting)
    else:
        init_pop_func = basic_init_pop_generator
        init_pop_params = BasicStartingLocationParams(initial_spread = 0.1)
    extra_info = {"Desc": "Basic NK landscape with N = {}, K = {}, popsize = {}, mutation_rate = {}".format(N, K, popsize, mutation_rate)}
    return EnvParams(mutation_generator = basic_mutation_generator, mutation_params = mut_params, 
                     landscape_generator = nk_land_generator, landscape_params = land_params, 
                     initial_pop_generator = init_pop_func, initial_pop_params = init_pop_params, 
                     extra_info = extra_info)
    
def get_basic_fixed_land_params(land, popsize: int, 
                                mutation_rate_normed: float, 
                                use_codon : bool = False, 
                                start_location = None, 
                                shuffle_fraction = None, 
                                mask_number = 0, 
                                multi_land: bool = False,
                                num_elites :int = 0
                                ) -> EnvParams:
    if use_codon:
        if not multi_land:
            assert land.shape[0] == 20, "If using codon, the first dimension must be 20, the number of amino acids."
        for i in land.shape[1:]:
            assert i == 20, "If using codon, all dimensions must be 20, the number of amino acids."
        num_options = 4
        if multi_land:
            gene_size = len(land.shape) * 3 - 3
        else:
            gene_size = len(land.shape) * 3
    else:
        num_options = land.shape[-1]
        if multi_land:
            gene_size = len(land.shape) - 1
        else:
            gene_size = len(land.shape)
        
    mutation_rate = mutation_rate_normed/(gene_size - mask_number)
    mut_params = BasicMutationParams(mutation_chance = mutation_rate, num_options = num_options, num_elites= num_elites)
    if shuffle_fraction is None:
        land_params = FixedLandParams(landscape = land, use_codon = use_codon, gene_size = gene_size, 
                                      popsize = popsize, mask_number=mask_number, multi_land=multi_land)
    else:
        land_params = FixedLandParams(landscape = land, use_codon = use_codon, gene_size = gene_size, 
                                      popsize = popsize, shuffle_landscape = True, shuffle_fraction = shuffle_fraction, mask_number=mask_number, multi_land=multi_land)
    if start_location is None:
        init_pop_params = BasicStartingLocationParams(initial_spread = 0.1)
        i_pop_gen = basic_init_pop_generator
    else:
        init_pop_params = BasicStartingLocationParams(fixed_start = True, starting_location = start_location)
        i_pop_gen = fixed_init_pop_generator
        

    extra_info = {"Desc": "Basic fixed landscape with popsize = {}, mutation_rate = {}".format(popsize, mutation_rate)}
    return EnvParams(mutation_generator = basic_mutation_generator, mutation_params = mut_params,
                     landscape_generator= fixed_land_generator, landscape_params = land_params,
                     initial_pop_generator= i_pop_gen, initial_pop_params = init_pop_params,
                     extra_info= extra_info)

@chex.dataclass
class BasicEvalParams:
    eval_start_step: int
    eval_end_step: int = -1
    
@chex.dataclass
class EnvEvalParams:
    num_landscapes: int
    num_starts_per_landscape: int
    num_reps_per_start: int
    num_steps: int
    eval_method: typing.Callable
    eval_method_params: typing.Any
    
    
def get_basic_eval_method(eval_params : BasicEvalParams):
    def to_ret(arr):
        return jnp.mean(arr[:, :, :, eval_params.eval_start_step:eval_params.eval_end_step])
    return to_ret
    
def get_pessimistic_eval_method(eval_params : BasicEvalParams):
    """
    Only cares about the worse case scenarion for a starting point. 
    """
    def to_ret(arr):
        return jnp.mean(jnp.min(jnp.mean(arr[:, :, :, eval_params.eval_start_step:eval_params.eval_end_step], axis = 3), axis = 2))
    return to_ret


@chex.dataclass
class Selection_Strat_Maker:
    selection_pair_getter : typing.Callable
    param_lower_limits : chex.Array
    param_upper_limits : chex.Array
    selection_pair_getter_extra_params : typing.Any = None
    chance_selection_pair_getter : typing.Callable = None # Gives us the function giving probabilities, rather than the selection funciton itself.

def get_single_eval_strategy_broad(strategy : Selection_Strat_Maker, 
                                   env_params : EnvParams , 
                                   eval_params: EnvEvalParams, 
                                   apply_norming : bool = False,  
                                   give_alldata : bool = False, 
                                   get_extra_data: bool = False, 
                                   extra_function_dict = None):
    """
    Gets a function which performs a single evaluation of the strategy,
    from the parameters given.
    """
    if apply_norming:
        normer = get_normer(strategy)
    else:
        normer = lambda x : x
    if give_alldata and not get_extra_data:
        eval_method = lambda x : x
    else:
        eval_method = eval_params.eval_method(eval_params.eval_method_params)
    def to_ret(rng, params):
        if strategy.selection_pair_getter_extra_params is not None:
            select_pair = strategy.selection_pair_getter(normer(params), strategy.selection_pair_getter_extra_params)
        else:
            select_pair = strategy.selection_pair_getter(normer(params))
        results = evaluate_strategy_broad(rng, select_pair, env_params, eval_params, get_extra_data, extra_function_dict)
        if get_extra_data:
            return results
        else:
            return eval_method(results)
    return jax.jit(to_ret)


def evaluate_strategy_broad(rng, selection_pair , env_params : EnvParams , eval_params : EnvEvalParams, get_extra_data : bool = False, extra_function_dict = None):
    """
    Evalutes a single strategy on a given set of params.
    Outputs a single array, of shape (landscapes, starts, reps, timesteps)
    """
    mut_method = env_params.mutation_generator(env_params)
    
    if extra_function_dict is None:
        extra_function_dict = evo_runs.default_extra_info

    def single_landscape(rng):
        #gives the results for a single landscape
        r1, r2 = jr.split(rng, 2)
        land_func = env_params.landscape_generator(r1, env_params)

        def single_start(rng):
            # Gives the resutls for a single start
            r1, r2 = jr.split(rng)
            i_pop = env_params.initial_pop_generator(r1, env_params, land_func)
            def single_rep(rng):
                # Gives the results of a single rep
                if get_extra_data:
                    return evo_runs.cont_evo_run_extras(rng, 
                                                        i_pop, 
                                                        selection_pair[1], 
                                                        selection_pair[0], 
                                                        mut_method, 
                                                        land_func, 
                                                        num_steps=eval_params.num_steps, 
                                                        num_elites= env_params.mutation_params.num_elites,
                                                        extra_function_dict=extra_function_dict,
                                                        )[1]
                else:
                    return evo_runs.cont_evo_run(rng,
                                                 i_pop,
                                                 selection_pair[1], 
                                                 selection_pair[0],
                                                 mut_method, 
                                                 land_func, 
                                                 num_elites= env_params.mutation_params.num_elites,
                                                 num_steps=eval_params.num_steps,
                                                 )[1]
            
            rand_keys = jr.split(r2, eval_params.num_reps_per_start)
            single_rep_vmapped = jax.jit(jax.vmap(single_rep))
            
            return single_rep_vmapped(rand_keys)
        
        rand_keys = jr.split(r2, eval_params.num_starts_per_landscape)
        sing_start_vmapped = jax.jit(jax.vmap(single_start))

        return sing_start_vmapped(rand_keys)
    
    rand_keys = jr.split(rng, eval_params.num_landscapes)
    single_land_vmapped = jax.jit(jax.vmap(single_landscape))

    return single_land_vmapped(rand_keys) 



@chex.dataclass
class NK_Env_Params:
    mutation_chance : float
    N : int
    K : int
    popsize : int
    num_steps : int 
    distribution :typing.Callable
    eval_lower_time : int = 80


def evaluate_strategy_nk(rng, selection_pair , nk_params : NK_Env_Params , num_reps : int, num_landscapes : int):
    """
    Evalutes a single strategy on a given NK model.
    Outputs a single array, of shape (num_landscapes, num_reps, timesteps)
    """
    mut_method = pop.get_basic_pop_mutation(nk_params.mutation_chance)

    def single_landscape(rng):
        #gives the mean performance of a single landscape
        r1, r2 = jr.split(rng, 2)
        land_func = ld.gen_NK_func(r1, nk_params.N, nk_params.K, dist=nk_params.distribution )

        def single_rep(rng):
            # Gives the mean performance of a single rep
            r1, r2 = jr.split(rng)
            i_pop = pop.gen_clustered_initial_population(r1, nk_params.N, nk_params.popsize, 0.1)
            return evo_runs.cont_evo_run(r2, i_pop, selection_pair[1], selection_pair[0], mut_method, land_func, num_steps=nk_params.num_steps)[1]
        
        rand_keys = jr.split(r2, num_reps)
        sing_rep_vmapped = jax.jit(jax.vmap(single_rep))

        return sing_rep_vmapped(rand_keys)
    
    single_land_vmapped = jax.jit(jax.vmap(single_landscape))

    return single_land_vmapped( jr.split(rng, num_landscapes))


@chex.dataclass
class Fixed_Land_Params:
    mutation_chance : float
    popsize : int
    num_steps : int 
    eval_lower_time : int = 80
    do_gb1_start : bool = False


import pkg_resources
with pkg_resources.resource_stream(__name__, 'data/GB1_landscape_array.pkl') as f:  
    GB1 = pickle.load(f)
    
# try:
#     with open('direvo/data/GB1_landscape_array.pkl', 'rb') as f:
#         GB1 = pickle.load(f)
# except FileNotFoundError:
#     # print current working directory
#     import os
#     print(os.getcwd())
#     raise FileNotFoundError("GB1_landscape_array.pkl not found. Make sure you're running this from the direvo directory")

GB1_landscape = ld.get_pre_defined_landscape_function(GB1)

def evaluate_strategy_fixed(rng, selection_pair , land_params : Fixed_Land_Params , num_reps : int, num_starts : int, landscape_arr = GB1):
    """
    Evalutes a single strategy on a given NK model.
    Outputs a single array, of shape (num_landscapes, num_reps, timesteps)
    """
    num_options = landscape_arr.shape[0]
    num_genes = len(landscape_arr.shape)
    landscape_func = ld.get_pre_defined_landscape_function(landscape_arr)
    mut_method = pop.get_basic_pop_mutation(land_params.mutation_chance, num_options)
    

    def single_start(rng):
        #gives the mean performance of a single landscape
        r1, r2 = jr.split(rng, 2)

        if land_params.do_gb1_start:
            i_pop = jnp.array([3,17,0,3])*jnp.ones((4,land_params.popsize), dtype=int).T
        else:
            i_pop = pop.gen_clustered_initial_population(r1, num_genes, land_params.popsize, 0.1)

        def single_rep(rng):
            # Gives the mean performance of a single rep
            return evo_runs.cont_evo_run(rng, i_pop, selection_pair[1], selection_pair[0], mut_method, landscape_func, num_steps=land_params.num_steps)[1]
        
        rand_keys = jr.split(r2, num_reps)
        sing_rep_vmapped = jax.jit(jax.vmap(single_rep))

        return sing_rep_vmapped(rand_keys)
    
    single_land_vmapped = jax.jit(jax.vmap(single_start))

    return single_land_vmapped( jr.split(rng, num_starts))



def atan_norming(x, low_lim = 0.0, upp_lim = 1.0):
    return low_lim + (upp_lim - low_lim) * (jnp.arctan(x)/jnp.pi + 0.5 )


def get_normer( strategy : Selection_Strat_Maker):
    n = strategy.param_lower_limits.shape
    has_lower_inf = strategy.param_lower_limits == -jnp.inf
    has_upper_inf = strategy.param_upper_limits == jnp.inf
    no_change_inds = jnp.where(has_lower_inf & has_upper_inf)[0]
    positive_exp_inds = jnp.where( (~has_lower_inf) & has_upper_inf)[0]
    negative_exp_inds = jnp.where( has_lower_inf & (~has_upper_inf) )[0]
    atan_norm_inds = jnp.where( (~has_lower_inf) & (~has_upper_inf))[0]
    def to_ret(a : chex.Array):
        outy = jnp.zeros(n)
        outy = outy.at[no_change_inds].set(a[no_change_inds])
        outy = outy.at[positive_exp_inds].set(strategy.param_lower_limits[positive_exp_inds] + jnp.exp(a[positive_exp_inds]))
        outy = outy.at[negative_exp_inds].set(strategy.param_upper_limits[negative_exp_inds] - jnp.exp(-a[negative_exp_inds]))
        outy = outy.at[atan_norm_inds].set( atan_norming(a[atan_norm_inds], strategy.param_lower_limits[atan_norm_inds], strategy.param_upper_limits[atan_norm_inds])  )     
        return outy
    return to_ret


def get_single_eval_strategy(strategy : Selection_Strat_Maker, nk_params : NK_Env_Params , num_reps : int, num_landscapes : int, apply_norming : bool = False,  give_alldata : bool = False):
    if apply_norming:
        normer = get_normer(strategy)
    else:
        normer = lambda x : x
    def to_ret(rng, params):
        select_pair = strategy.selection_pair_getter(normer(params))
        if give_alldata:
            return evaluate_strategy_nk(rng, select_pair, nk_params, num_reps, num_landscapes)
        else:
            return jnp.mean(evaluate_strategy_nk(rng, select_pair, nk_params, num_reps, num_landscapes)[:,:,nk_params.eval_lower_time : ])
    return to_ret


def get_single_eval_strategy_fixed(strategy : Selection_Strat_Maker, land_params : Fixed_Land_Params , num_reps : int, num_starts : int, apply_norming : bool = False,  give_alldata : bool = False):
    if apply_norming:
        normer = get_normer(strategy)
    else:
        normer = lambda x : x
    def to_ret(rng, params):
        select_pair = strategy.selection_pair_getter(normer(params))
        if give_alldata:
            return evaluate_strategy_fixed(rng, select_pair, land_params, num_reps, num_starts)
        else:
            return jnp.mean(evaluate_strategy_fixed(rng, select_pair, land_params, num_reps, num_starts)[:,:,land_params.eval_lower_time : ])
    return to_ret


def sigmoid_pairgetter(params):
    sigmoid_params = dict(zip(["threshold", "base_chance", "steepness"], params))
    return pop.select_cells( pop.sigmoid_select, sigmoid_params)

def spiking_pairgetter(params):
    """
    Returns the spiking selection pair.
    Takes in a 2 element array, the first element is the threshold, the second is the base chance.
    """
    spiking_params = dict(zip(["threshold", "base_chance"], params))
    return pop.select_cells(pop.spiking_select, spiking_params)

def raw_sigmoid_pairgetter(params):
    sigmoid_params = dict(zip(["threshold", "base_chance", "steepness"], params))
    return pop.sigmoid_select, sigmoid_params

def raw_spiking_pairgetter(params):
    spiking_params = dict(zip(["threshold", "base_chance"], params))
    return pop.spiking_select, spiking_params

def multispiking_pairgetter(params):
    n = params.shape[0]
    spiking_params = {}
    spiking_params["min chance"] = params[0]
    spiking_params["max chance"] = params[1]
    spiking_params["step thresholds"] = params[2:n // 2 + 1]
    spiking_params["step relative changes"] = params[n // 2 + 1:]
    return pop.select_cells(pop.multi_spiking_select,spiking_params)

def raw_multispiking_pairgetter(params):
    n = params.shape[0]
    spiking_params = {}
    spiking_params["min chance"] = params[0]
    spiking_params["max chance"] = params[1]
    spiking_params["step thresholds"] = params[2:n // 2 + 1]
    spiking_params["step relative changes"] = params[n // 2 + 1:]
    return (pop.multi_spiking_select,spiking_params)

def multispiking_pairgetter_with_delay(params):
    n = params.shape[0]
    spiking_params = {}
    spiking_params["min chance"] = params[0]
    spiking_params["max chance"] = params[1]
    spiking_params["step thresholds"] = params[2:n // 2 + 1]
    spiking_params["step relative changes"] = params[n // 2 + 1:]
    select_strat = pop.select_cells(pop.multi_spiking_select,spiking_params)
    greedy_strat = pop.select_cells(pop.spiking_select, {"threshold": 0.8, "base_chance" : 0.0})
    return pop.alternating_strategy(greedy_strat, select_strat, 80, 500)

def raw_multispiking_pairgetter_with_delay(params):
    n = params.shape[0]
    spiking_params = {}
    spiking_params["min chance"] = params[0]
    spiking_params["max chance"] = params[1]
    spiking_params["step thresholds"] = params[2:n // 2 + 1]
    spiking_params["step relative changes"] = params[n // 2 + 1:]
    return [ (pop.multi_spiking_select,spiking_params), (pop.spiking_select, {"threshold": 0.8, "base_chance" : 0.0})]


sigmoid_strategy = Selection_Strat_Maker(selection_pair_getter = sigmoid_pairgetter, param_lower_limits = jnp.array([0., 0., 0.]), param_upper_limits = jnp.array([1., 1., jnp.inf]), 
                                         chance_selection_pair_getter=raw_sigmoid_pairgetter)
spiking_strategy = Selection_Strat_Maker(selection_pair_getter = spiking_pairgetter, param_lower_limits = jnp.array([0., 0.]), param_upper_limits = jnp.array([1., 1.]),
                                         chance_selection_pair_getter=raw_spiking_pairgetter)

def flow_strategy_pairgetter(params):
    return pop.flow_select_cells(params)

flow_strategy = Selection_Strat_Maker(selection_pair_getter = flow_strategy_pairgetter, param_lower_limits = jnp.array([0., 0.]), param_upper_limits = jnp.array([1., 1.]),
                                             chance_selection_pair_getter= None)

def wrapped_pairgetter(params, extra_params):
    wrapped_strat = extra_params["wrapped strat"]
    selec_func, selec_params = wrapped_strat.chance_selection_pair_getter(params)

    wrappy_params = {"wrapped function": selec_func, "wrapped params" : selec_params, "weighting" : extra_params["weighting"]}
    
    return pop.select_cells(pop.multi_select_wrapper, wrappy_params)

def wrapped_raw_pairgetter(params, extra_params):
    wrapped_strat = extra_params["wrapped strat"]
    selec_func, selec_params = wrapped_strat.chance_selection_pair_getter(params)

    wrappy_params = {"wrapped function": selec_func, "wrapped params" : selec_params, "weighting" : extra_params["weighting"]}
    
    return (pop.multi_select_wrapper, wrappy_params)

def get_multi_select_wrapper(wrapped_strategy : Selection_Strat_Maker, weighting = jnp.array([1., 1.])):
    
    return Selection_Strat_Maker(selection_pair_getter = wrapped_pairgetter, param_lower_limits = wrapped_strategy.param_lower_limits, param_upper_limits = wrapped_strategy.param_upper_limits,
                                 chance_selection_pair_getter=wrapped_raw_pairgetter, selection_pair_getter_extra_params={"wrapped strat" : wrapped_strategy, "weighting" : weighting})


def get_multispiking_strategy(num_of_jumps : int = 1):
    n = 2 + 2*num_of_jumps
    lowlim_arr = jnp.zeros(n)
    upplim_arr = jnp.ones(n)
    return Selection_Strat_Maker(selection_pair_getter = multispiking_pairgetter, 
                                 param_lower_limits = lowlim_arr, 
                                 param_upper_limits = upplim_arr,
                                 chance_selection_pair_getter= raw_multispiking_pairgetter)
    




def sel_pgetter(params):
    n = params.shape[0]
    time_1 = jnp.round(params[0])
    time_2 = jnp.round(params[1])
    select_1 = multispiking_pairgetter(params[2: n//2 +1])
    select_2 = multispiking_pairgetter(params[n//2 +1:])
    return pop.alternating_strategy(select_1, select_2, time_1, time_2)

def raw_sel_pgetter(params):
    n = params.shape[0]
    select_1 = raw_multispiking_pairgetter(params[2: n//2 +1])
    select_2 = raw_multispiking_pairgetter(params[n//2 +1:])
    return [select_1, select_2]

def get_alternating_multispiking_strategy_old(jump_num1 : int = 1, jump_num2 : int = 1):
    n = 6 + 2*(jump_num1 + jump_num2)
    lowlim_arr = jnp.zeros(n)
    lowlim_arr = lowlim_arr.at[:2].set(1.0)
    upplim_arr = jnp.ones(n)
    upplim_arr = upplim_arr.at[:2].set(jnp.inf)
    return Selection_Strat_Maker(selection_pair_getter= sel_pgetter, 
                                 param_lower_limits=lowlim_arr,
                                 param_upper_limits=upplim_arr,
                                 chance_selection_pair_getter=raw_sel_pgetter)

def get_alternating_multispiking_strategy(jump_num1 : int = 1, jump_num2 : int = 1):
    n = 6 + 2*(jump_num1 + jump_num2)
    lowlim_arr = jnp.zeros(n)
    lowlim_arr = lowlim_arr.at[:2].set(1.0)
    upplim_arr = jnp.ones(n)
    upplim_arr = upplim_arr.at[:2].set(100.0)
    return Selection_Strat_Maker(selection_pair_getter= sel_pgetter, 
                                 param_lower_limits=lowlim_arr,
                                 param_upper_limits=upplim_arr,
                                 chance_selection_pair_getter=raw_sel_pgetter)

def get_forced_delay_msk_strategy(num_of_jumps : int = 2):
    n = 2 + 2*num_of_jumps
    lowlim_arr = jnp.zeros(n)
    upplim_arr = jnp.ones(n)
    return Selection_Strat_Maker(selection_pair_getter = multispiking_pairgetter_with_delay, 
                                 param_lower_limits = lowlim_arr, 
                                 param_upper_limits = upplim_arr,
                                 chance_selection_pair_getter= raw_multispiking_pairgetter_with_delay)

def array_to_multispike_dict(params):
    n = params.shape[0]
    spiking_params = {}
    spiking_params["min chance"] = params[0]
    spiking_params["max chance"] = params[1]
    spiking_params["step thresholds"] = params[2:n // 2 + 1]
    spiking_params["step relative changes"] = params[n // 2 + 1:]
    return spiking_params
