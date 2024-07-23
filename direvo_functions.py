import jax
import jax.numpy as jnp
import jax.random as jr

### DIRECTED EVOLUTION ITERATOR -------------------------------------------------------------------------------------

default_extra_info = {
    "fitness": lambda *args, **kwargs : kwargs["fitnesses"],
    "pop": lambda *args, **kwargs : kwargs["pop"]
}

def run_directed_evolution(rng,
                        i_pop,
                        selection_function,
                        mutation_function, 
                        fitness_function, 
                        fitness_noise = 1e-4,
                        num_steps=30, 
                        extra_function_dict = default_extra_info):
    
    """
    Function to run a directed evolution simulation.

    Parameters:
    - rng: Jax random number key (e.g jax.random.PRNGKey(0)).
    - i_pop: Initial starting population. An array of dimensions (popsize, N).
    - selection_function: Generated from build_selection_function. Takes in a population and output true/false values of selection.
    - mutation_function: Genreated from build_mutation_function. Takes in a population and applies mutation.
    - fitness_function: Generated from build_NK_landscape_function/build_empirical_function. Takes in population sequences and outputs fitness values.
    - fitness_noise: Noise to be applied to fitness values.
    - num_steps: The number of mutation-selection iterations to be performed.
    - output_dict: Dictionary of information to be outputted.

    Returns:
    - (Customisable) dictionary of populations and fitness values.
    """
    
    ######################################
    ## Function for a single iteration. ##
    ######################################
    
    def single_iteration(rng,
                         population, 
                         selection_function, 
                         mutation_function, 
                         fitness_function, 
                         fitness_noise=1e-4,
                         extra_function_dict = {}):

        r1, r2, r3 = jr.split(rng, 3)
        
        # Get population fitness values.
        population_fitness = fitness_function(population) # Calculate population fitness values.
        fitness_noise =  population_fitness*jr.normal(r1, population_fitness.shape)*fitness_noise # Add noise to fitness readings.
        population_fitness = population_fitness + fitness_noise

        # Perform selection on the population.
        selected_population = selection_function(r2, population_fitness)
        resamped_pop = population[selected_population]

        # Apply mutation.
        mutated_pop = mutation_function(r3, resamped_pop)
        
        current_info_dict = {"fitnesses" : population_fitness, "pop" : population}
        extra_info = {key: func(**current_info_dict) for key, func in extra_function_dict.items() }
        return mutated_pop, extra_info

    ###############################################################
    ## Multiple iterations with Jax lax scan for GPU efficiency. ##
    ###############################################################

    def multiple_iterations(population, local_rng):

        new_pop, extra_info = single_iteration(local_rng, 
                                   population, 
                                   selection_function,
                                   mutation_function, 
                                   fitness_function,
                                   fitness_noise=fitness_noise,
                                   extra_function_dict = extra_function_dict)
        
        return ((new_pop) , extra_info)
    
    rng_array = jr.split(rng, num_steps)

    return jax.lax.scan(multiple_iterations,(i_pop),rng_array)


### LANDSCAPE GENERATION -------------------------------------------------------------------------------------------------

def build_empirical_landscape_function(landscape):
    """
    Looks up fitness values on pre-defined landscape such as GB1.

    Parameters:
    - landscape: N-dimensional empirical landscape array. (e.g. GB1 shape 20x20x20x20).

    Returns:
    - Function that takes GB1 sequence as input, and returns fitness value.
    """

    def get_fitness(i):
        return landscape[tuple(i)]

    return jax.jit(jax.vmap(get_fitness))


def build_NK_landscape_function(rng, N, K, fitness_distribution=jr.normal):
    """
    Looks up fitness values on an NK landscape.

    Parameters:
    - rng: Jax random number key (e.g jax.random.PRNGKey(0)).
    - N: integer value representing number of sites in the gene.
    - K: integer value representing the number of interactions per site.
    - fitness_distribution: Distribution from which individual fitness values are sampled.

    Returns:
    - Function that takes NK gene sequence as input, and returns fitness value.
    """

    r1, r2, r3 = jr.split(rng, 3)

    ##################################
    ## Generate interaction matrix. ##
    ##################################

    base_row = 1* (jnp.arange(N-1) < K) # Array of size N, with K entries = 1.

    def permutate_rows(rng, i):
        perm_row = jr.permutation(rng, base_row)
        return jnp.insert(perm_row, i, 1.0)
    
    permutate_rows = jax.vmap(permutate_rows)
    interaction_matrix = permutate_rows(jr.split(r1, N), jnp.arange(N))

    ##########################################
    ## Build function for sampling from NK. ##
    ##########################################
    
    # Function for generating rng keys, ensuring that they are derived from the same base rng.
    vector_foldin = jax.vmap(lambda base_rng, data: jr.fold_in(base_rng, data))
    
    fitness_distribution = jax.vmap(fitness_distribution)

    def get_fitness(gene):
        individual_site_fitness = vector_foldin(jr.split(r2, N), gene)
        interaction_fitness = (interaction_matrix @ individual_site_fitness) + jr.split(r3, N)
        return jnp.sum(fitness_distribution(interaction_fitness))
    
    return jax.jit(jax.vmap(get_fitness))


### MUTATION FUNCTION ---------------------------------------------------------------------------------------------

def build_mutation_function(mutation_chance, num_options=2):
    """
    Generates a function that mutates a population of gene sequence.

    Parameters:
    - mutation_chance: Probability of a mutation per site.
    - num_options: Number of possibilities each site can be. Defaults to 2 for [0,1] options.

    Returns:
    - Function that applies mutations to a whole population of gene sequences.
    """

    def mutation_function(rng, pop):
        r1, r2 = jr.split(rng, 2)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)
        mut_delta = jr.randint(r2, pshape, 1, num_options)
        return (pop + (has_mutation*mut_delta)) % num_options
    return mutation_function

### SELECTION FUNCTION ---------------------------------------------------------------------------------------------

def build_selection_function(selection_function_shape, params):
    """
    Function for determining which cells of a population are selected.

    Parameters:
    - selection_function: A function that takes in an array of fitness values, and outputs an array of probabilities of selection.
                          Options in selection_function_library.
    - params: Dictionary containing the parameters defining the shape of the selection_function.

    Returns:
    - Array of true/false values describing which population members are selected.
    """

    def selection_function(rng, fitnesses, state=0):
        psize = fitnesses.shape[-1]
        selection_prob = selection_function_shape(fitnesses, params)
        selected = jnp.ones((psize,))*jr.bernoulli(rng,
                                                   p=selection_prob, shape=(psize,))

        return jr.choice(rng, jnp.arange(psize), (psize,), p=selected)
    
    return selection_function

