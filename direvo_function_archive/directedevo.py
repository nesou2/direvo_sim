import jax
import jax.numpy as jnp
import jax.random as jr

def get_quad_matrix(key, N, K, mean_diag, sig_diag, mean_int, sig_int):
    """
    This is a matrix for what I thought the NK model was.
    It's kinda dumb.
    It involves a linear component (the diagonal) and random `interactions'
    with, on average, K other things. Each interaction is additive,
    so the whole thing can be represented by this inner product matrix
    """
    k1, k2, k3 = jr.split(key, 3)
    self_int = jnp.diag(jr.normal(k1, (N,)) * sig_diag) + mean_diag
    other_int_onehot = jr.bernoulli(k2, K/ (2*N -2), (N,N))
    other_int_onehot = other_int_onehot * (1 - jnp.identity(N))
    return self_int + other_int_onehot * (jr.normal(k3, (N,N)) * sig_int + mean_int)

def get_fit_func_from_mat(mat):
    """
    gets the corrosponding function from a matrix. very simple
    """
    return lambda x : jnp.sum((x @ mat) * x, axis = 1)



def get_independent_noise_func(rng_baseline, N):
    """
    Gets a random function from {0,1}^N,
    or the iid normal distribution over this space.
    """
    mapper_vect = jnp.power(2, jnp.arange(N))
    def get_noise(v):
        noise_int = jnp.dot(mapper_vect, v)
        new_key = jr.fold_in(rng_baseline, noise_int)
        return jr.normal(new_key)
    return jax.vmap(get_noise)

def get_NK_int_matrix(rng, N,K):
    """
    gets a random interaction matrix for the NK model
    """
    base_row = jnp.array([1.0] * K + [0.0]*(N - K - 1))
    def get_row_i(rng, i):
        perm_row = jr.permutation(rng, base_row)
        return jnp.insert(perm_row, i, 1.0)
    get_rowz = jax.vmap(get_row_i)
    return get_rowz(jr.split(rng, N), jnp.arange(N))

def get_NK_func_from_matrix(rng_baseline, int_matrix, dist = jr.normal):
    """
    Given a matrix of interactions,
    gives the corrosponding function for the fitness landscape
    """
    N = int_matrix.shape[0]
    rng_bases = jr.split(rng_baseline, N)
    mapper_vect = jnp.power(2, jnp.arange(N))
    vector_foldin = jax.vmap(lambda key, data: dist(jr.fold_in(key,data)) )
    def get_noise(v):
        masked_int = int_matrix * v
        seedies = jnp.dot( masked_int, mapper_vect)
        return jnp.sum(vector_foldin(rng_bases, seedies))
    return jax.vmap(get_noise)

def gen_clustered_inital_population(rng, N, popsize, expected_diff):
    """
    Generations an initial, clusteded population.
    Generates a point at random, then adds noise such that you
    have, on average, expected_diff mutations
    """
    r1, r2 = jr.split(rng)
    start_gene = jr.bernoulli(r1, 0.5, (N,))
    p = expected_diff/N
    mutations = jr.bernoulli(r2, p, (popsize,N))
    return (mutations + start_gene)%2

def gen_NK_func(rng, N, K, dist = jr.normal):
    """
    Puts the sub-functions together 
    to generate the NK function for the fitness landscape
    """
    r1, r2 = jr.split(rng)
    matty = get_NK_int_matrix(r1, N, K)
    return get_NK_func_from_matrix(r2, matty, dist = dist)

def gen_simple_land_func(rng, N, lin_scale, indp_scale):
    """
    Generates a simple landscape
    of a linear comenent , line_scale
    and a iid component, indp_scale
    """
    r1, r2 = jr.split(rng)
    lin_comp = lin_scale * jr.normal(rng, (N,))
    rand_comp = get_independent_noise_func(r2, N)
    return lambda v : jnp.dot(lin_comp, v) + indp_scale * rand_comp(v)

def run_dir_evo_iter(key, fitness_func, population, fit_shape_func, mut_chance):
    """
    Runs a single iteraction of fitness, selection, mutation
    Can take in an arbritary fitness landscape

    key: rng key
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    population: the inital population
    fit_shape_func: the function that transforms the vector of fitness values into
    probabilities
    mut_chance: the chance that any inidivdual gene mutates.
    """
    k1, k2 = jr.split(key)
    fitties = fitness_func(population)
    weights = fit_shape_func(fitties)
    resamped_pop = jr.choice(k1, population, weights.shape, p = weights )
    mutated_pop = (resamped_pop + jr.bernoulli(k2, mut_chance, population.shape ))%2
    return mutated_pop

def do_dir_evo_run(rng, i_pop, fitness_func, fit_shape_func, mut_chance, num_steps = 10 ):
    """
    Runs several iteration of fitness, selection, mutation

    
    key: rng key
    fitness_func: the fitness landscape to use. Needs to be pre-vmapped,
    aka it needs to take the entire population at once
    population: the inital population
    fit_shape_func: the function that transforms the vector of fitness values into
    probabilities
    mut_chance: the chance that any inidivdual gene mutates.
    """
    def to_scan(pop, local_rng):
        new_pop = run_dir_evo_iter(local_rng, fitness_func, pop, fit_shape_func, mut_chance)
        return (new_pop, jnp.max(fitness_func(new_pop)) )
    keyz = jr.split(rng, num_steps)
    return jax.lax.scan(to_scan, i_pop, keyz)





