import jax
import jax.numpy as jnp
import jax.random as jr



def cont_evo_iter(rng, 
                  population, 
                  selection_state,  
                  selection_function, 
                  mutation_method, 
                  fitness_func, 
                  add_noise = True,
                  num_elites = 0,):
    """
    Runs a single iteraction of fitness, selection, mutation
    Can take in an arbritary fitness landscape
    rng: rng key
    population: the inital population
    mut_chance: the chance that any inidivdual gene mutates.
    selection_function: Takes the fitnesses, and outputs the indicies of the
    new population to take.
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    """
    k1, k2, k3 = jr.split(rng, 3)
    pop_size = population.shape[0]
    fitties = fitness_func(population)
    if add_noise:
        fitness_noise =  fitties*jr.normal(k3, fitties.shape)*1e-4
    else:
        fitness_noise = jnp.zeros(fitties.shape)
    fitties = fitties + fitness_noise
    pop_indicies, new_selection_state = selection_function(k1, fitties, selection_state)
    
    resamped_pop = population[pop_indicies]
    mutated_pop = mutation_method(k2, resamped_pop)
    
    elite_locations = 1*(jnp.arange(pop_size) < num_elites)
    sorted_population = population[jnp.argsort(-fitties)]
    pop_with_elites = jnp.array([mutated_pop, sorted_population])[elite_locations, jnp.arange(pop_size), :]
    
   
    return pop_with_elites, new_selection_state


def cont_evo_run(rng,
                 i_pop, 
                 i_sel_state, 
                 selection_function, 
                 mutation_method, 
                 fitness_func, 
                 num_steps=30, 
                 summary_metric=jnp.max,
                 num_elites = 0,):
    """
    Runs several iteration of fitness, selection, mutation
    key: rng key
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    population: the inital population
    selection_function: the function that takes in an array of fitnesses, and gives the 
    indicies of the population to keep
    mut_chance: the chance that any inidivdual gene mutates.
    """
    def to_scan(state_pair, local_rng):
        pop, old_selection_state = state_pair
        new_pop, new_selection_state =\
              cont_evo_iter(local_rng, pop, old_selection_state, selection_function, mutation_method, fitness_func, add_noise = True, num_elites = num_elites)
        return ( (new_pop, new_selection_state) , summary_metric(fitness_func(new_pop)))
    keyz = jr.split(rng, num_steps)
    return jax.lax.scan(to_scan, (i_pop, i_sel_state) , keyz)


def cont_evo_iter_extras(rng,
                         population, 
                         selection_state,  
                         selection_function, 
                         mutation_method, 
                         fitness_func, 
                         add_noise = True, 
                         num_elites = 0,
                         extra_function_dict = {}):
    """
    Runs a single iteraction of fitness, selection, mutation
    Gives extra information above the default
    Can take in an arbritary fitness landscape
    rng: rng key
    population: the inital population
    mut_chance: the chance that any inidivdual gene mutates.
    selection_function: Takes the fitnesses, and outputs the indicies of the
    new population to take.
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    """
    k1, k2, k3 = jr.split(rng, 3)
    fitties = fitness_func(population)
    if add_noise:
        fitness_noise =  fitties*jr.normal(k3, fitties.shape)*1e-4
    else:
        fitness_noise = jnp.zeros(fitties.shape)
    fitties = fitties + fitness_noise
    pop_indicies, new_selection_state = selection_function(k1, fitties, selection_state)
    resamped_pop = population[pop_indicies]
    mutated_pop = mutation_method(k2, resamped_pop)
    
    pop_size = population.shape[0]
    elite_locations = 1*(jnp.arange(pop_size) < num_elites)
    sorted_pop_ind = jnp.argsort(-fitties)
    sorted_population = population[sorted_pop_ind]
    
    pop_with_elites = jnp.array([mutated_pop, sorted_population])[elite_locations, jnp.arange(pop_size), :]
    
    pop_indicies_with_elites = jnp.array([pop_indicies, sorted_pop_ind])[elite_locations, jnp.arange(pop_size)]
    
    
    current_info_dict = {"fitnesses" : fitties, "pop" : population, "noise" : fitness_noise, "indicies" : pop_indicies_with_elites}
    #extra_info = {"indicies" : pop_indicies, "fitnesses" : fitties}
    extra_info = {key: func(**current_info_dict) for key, func in extra_function_dict.items() }
    return pop_with_elites, new_selection_state, extra_info


default_extra_info = {
    "fitness": lambda *args, **kwargs : kwargs["fitnesses"],
    "pop": lambda *args, **kwargs : kwargs["pop"],
    "indicies": lambda *args, **kwargs : kwargs["indicies"],
}

def cont_evo_run_extras(rng,
                        i_pop,
                        i_sel_state,
                        selection_function,
                        mutation_method, 
                        fitness_func, 
                        num_steps=30, 
                        add_noise = True,
                        num_elites = 0,
                        extra_function_dict = default_extra_info):
    """
    Runs several iteration of fitness, selection, mutation
    Gives extra, more detialed, info
    key: rng key
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    population: the inital population
    selection_function: the function that takes in an array of fitnesses, and gives the 
    indicies of the population to keep
    mut_chance: the chance that any inidivdual gene mutates.
    """
    def to_scan(state_pair, local_rng):
        pop, old_selection_state = state_pair
        new_pop, new_selection_state, extra_info =\
              cont_evo_iter_extras(local_rng, 
                                   pop,
                                   old_selection_state, 
                                   selection_function,
                                   mutation_method, 
                                   fitness_func,
                                   add_noise = add_noise, 
                                   num_elites=num_elites,
                                   extra_function_dict = extra_function_dict)
        return ( (new_pop, new_selection_state) , extra_info )
    keyz = jr.split(rng, num_steps)
    return jax.lax.scan(to_scan, (i_pop, i_sel_state) , keyz)




