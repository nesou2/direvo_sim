"""
Contain the functions relevent for the fitness landscapes
which we test/optimize over. 
"""

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
    other_int_onehot = jr.bernoulli(k2, K / (2*N - 2), (N, N))
    other_int_onehot = other_int_onehot * (1 - jnp.identity(N))
    return self_int + other_int_onehot * (jr.normal(k3, (N, N)) * sig_int + mean_int)


def get_fit_func_from_mat(mat):
    """
    gets the corrosponding function from a matrix. very simple
    """
    return lambda x: jnp.sum((x @ mat) * x, axis=1)


def get_independent_noise_func(rng_baseline, N):
    """
    Gets a random function from {0,1}^N,
    or the iid normal distribution over this space.
    """
    r1, r2 = jr.split(rng_baseline)
    mapper_vect = jr.split(r1, N)

    def get_noise(v):
        noise_int = mapper_vect * v.reshape((N, 1))
        new_key = jnp.sum(noise_int, axis = 0) + r2
        return jr.normal(new_key)
    return jax.vmap(get_noise)


def get_NK_int_matrix(rng, N, K):
    """
    gets a random interaction matrix for the NK model
    """
    #base_row = jnp.array([1.0] * K + [0.0]*(N - K - 1))
    base_row = 1* (jnp.arange(N-1) < K)

    def get_row_i(rng, i):
        perm_row = jr.permutation(rng, base_row)
        return jnp.insert(perm_row, i, 1.0)
    get_rowz = jax.vmap(get_row_i)
    return get_rowz(jr.split(rng, N), jnp.arange(N))


def get_NK_func_from_matrix(rng_baseline, int_matrix, dist=jr.normal):
    """
    Given a matrix of interactions,
    gives the corrosponding function for the fitness landscape
    Now also works for more than just 2 sites.
    """
    N = int_matrix.shape[0]

    vector_foldin = jax.vmap(lambda key, data: jr.fold_in(key, data))
    
    r1, r2 = jr.split(rng_baseline, 2)
    pre_scrable_vector = jr.split(r1, N)
    post_scramble_vector = jr.split(r2, N)
    
    vmap_dist = jax.vmap(dist)

    def get_noise(v):
        v_scramble = vector_foldin(pre_scrable_vector, v)
        sited_noise = (int_matrix @ v_scramble) + post_scramble_vector
        return jnp.sum(vmap_dist(sited_noise))
    return jax.jit(jax.vmap(get_noise))


def get_seb_NK_func_from_matrix(rng_baseline, int_matrix, dist=jr.normal):
    """
    My own idea for how to make the optimization landscape
    more interesting. The first few sites dictate the 
    fitness scale, and landscape `seed`
    So you need to optimize the fitness scale to actually determine
    if the overall function is good or not. I also 
    need to add more holes, to make getting something accidentally
    good much rarer. 
    """
    N = int_matrix.shape[0]

    vector_foldin = jax.vmap(lambda key, data: jr.fold_in(key, data))
    
    r1, r2 = jr.split(rng_baseline, 2)
    pre_scrable_vector = jr.split(r1, N)
    post_scramble_vector = jr.split(r2, N)
    
    vmap_dist = jax.vmap(dist)

    def get_noise(v):
        v_scramble = vector_foldin(pre_scrable_vector, v)
        sited_noise = (int_matrix @ v_scramble) + post_scramble_vector
        return jnp.sum(vmap_dist(sited_noise))
    return jax.jit(jax.vmap(get_noise))

def get_NK_func_from_matrix_old(rng_baseline, int_matrix, dist=jr.normal):
    """
    Given a matrix of interactions,
    gives the corrosponding function for the fitness landscape
    BUGGY AND OLD! 
    """
    N = int_matrix.shape[0]
    rng_bases = jr.split(rng_baseline, N)
    mapper_vect = jnp.power(2, jnp.arange(N)) # Suffers from overflow issues
    vector_foldin = jax.vmap(lambda key, data: dist(jr.fold_in(key, data)))

    def get_noise(v):
        masked_int = int_matrix * v
        seedies = jnp.dot(masked_int, mapper_vect)
        return jnp.sum(vector_foldin(rng_bases, seedies))
    return jax.vmap(get_noise)


def gen_NK_func(rng, N, K, dist=jr.normal):
    """
    Puts the sub-functions together 
    to generate the NK function for the fitness landscape
    """
    r1, r2 = jr.split(rng)
    matty = get_NK_int_matrix(r1, N, K)
    return get_NK_func_from_matrix(r2, matty, dist=dist)


def gen_simple_land_func(rng, N, lin_scale, indp_scale):
    """
    Generates a simple landscape
    of a linear comenent , line_scale
    and a iid component, indp_scale
    """
    r1, r2 = jr.split(rng)
    lin_comp = lin_scale * jr.normal(rng, (N,))
    rand_comp = get_independent_noise_func(r2, N)
    return lambda v: jnp.dot(lin_comp, v) + indp_scale * rand_comp(v)


def get_pre_defined_landscape_function(landscape):
    """
    Looks up fitness values on pre-defined landscape
    """
    def get_fitness(i):
        return landscape[tuple(i)]

    return jax.jit(jax.vmap(get_fitness ))



if False:
    None
    # Cdoef or getting the codon mapping. Putting it here for now
    # with open("AAdict.pkl", "rb") as f:
    #     acid_to_num_dict = pickle.load(f)
        
    # acid_to_num_dict["_"] = -1 

    # genetic_code = {
    #     'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    #     'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    #     'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    #     'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    #     'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    #     'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    #     'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    #     'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    #     'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    #     'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    #     'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    #     'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    #     'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    #     'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    #     'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    #     'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    # }


    # def codon_to_amino_acid(codon):
    #     codon = codon.upper()
    #     if codon in genetic_code:
    #         return genetic_code[codon]
    #     else:
    #         raise ValueError("Invalid codon: '{}'".format(codon))

    # # Example usage:
    # codon = 'ATG'
    # amino_acid = codon_to_amino_acid(codon)
    # print(f"The amino acid for codon {codon} is {amino_acid}")



    # base_pairs = ['T', 'C', 'A', 'G']
    # triples_list = [  [ [ genetic_code[a2 + a0 + a1] for a0 in base_pairs ]  for a1 in base_pairs ]  for a2 in base_pairs ]   

    # triples_list_convy = [  [ [ acid_to_num_dict[genetic_code[a2 + a0 + a1]] for a0 in base_pairs ]  for a1 in base_pairs ]  for a2 in base_pairs ]  


# GB1_start = baseys = "GTC", "GAC", "GGT", "GTT"
# Also, in numbers, numerical_baseys = [3, 0, 0, 3, 0, 2, 3, 0, 3, 3, 0, 0]

# Also, [3,17,0,3] is the amino acid sequence for GB1
GB1_acid_start = jnp.array([3,17,0,3])
GB1_codon_start = jnp.array([3, 0, 0, 3, 0, 2, 3, 0, 3, 3, 0, 0])

CODON_MAPPER = jnp.array([[[8, 18, 9, 7], [8, 18, 9, 7], [4, 18, -1, -1], [4, 18, -1, 10]],
 [[4, 1, 11, 13], [4, 1, 11, 13], [4, 1, 14, 13], [4, 1, 14, 13]],
 [[5, 19, 15, 18], [5, 19, 15, 18], [5, 19, 12, 13], [6, 19, 12, 13]],
 [[3, 2, 17, 0], [3, 2, 17, 0], [3, 2, 16, 0], [3, 2, 16, 0]]] )

INVERSE_CODON_MAPER = jnp.array([[3, 0, 3],
       [1, 0, 1],
       [3, 0, 1],
       [3, 0, 0],
       [0, 2, 0],
       [2, 0, 0],
       [2, 3, 0],
       [0, 0, 3],
       [0, 0, 0],
       [0, 0, 2],
       [0, 3, 3],
       [1, 0, 2],
       [2, 2, 2],
       [1, 0, 3],
       [1, 2, 2],
       [2, 0, 2],
       [3, 2, 2],
       [3, 0, 2],
       [0, 0, 1],
       [2, 0, 1]], dtype=jnp.int32)



def inverse_codon(codon):
    """
    Returns the inverse of a codon
    """
    indy = jnp.where( CODON_MAPPER == codon)
    return jnp.array([indy[0][0], indy[1][0], indy[2][0]])


def get_pre_defined_landscape_function_with_codon(landscape):
    """
    Looks up fitness values on pre-defined landscape which is 
    defined by amino acids. It used the codon look-up method.
    We also assume landscape has each dimension size 20, and doens't
    account for stop codons, which we map to being the min fitness.
    """
    n = len(landscape.shape)
    
    min_fitness = jnp.min(landscape)
    
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)
    
    def get_codon(i):
        return CODON_MAPPER[tuple(i)]
    
    vmapped_get_codons = jax.jit(jax.vmap(get_codon))
    
    def get_fitties(params):
        parries_reshaped = jnp.reshape(params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]
    
    return jax.jit(jax.vmap(get_fitties))


def get_pd_landscape_function_codon_masked(landscape, mask, replacment):
    """
    Looks up fitness values on pre-defined landscape which is 
    defined by amino acids. It used the codon look-up method.
    We also assume landscape has each dimension size 20, and doens't
    account for stop codons, which we map to being the min fitness.
    """
    n = len(landscape.shape)
    
    min_fitness = jnp.min(landscape)
    
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)
    
    def get_codon(i):
        return CODON_MAPPER[tuple(i)]
    
    vmapped_get_codons = jax.jit(jax.vmap(get_codon))
    
    def get_fitties(params):
        new_params = jnp.where(mask, replacment, params)
        parries_reshaped = jnp.reshape(new_params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]
    
    return jax.jit(jax.vmap(get_fitties))

def get_pd_landscape_function_masked(landscape, mask, replacment):
    """
    Looks up fitness values on pre-defined landscape
    """
    def get_fitness(i):
        new_i = jnp.where(mask, replacment, i)
        return landscape[tuple(new_i)]

    return jax.jit(jax.vmap(get_fitness ))


