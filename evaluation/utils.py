"""
A collection of utility functions for use in the Direvo package.
"""

import jax.numpy as jnp
import numpy as np

def function_map(func_list):
    def to_ret(*args, **kwargs):
        return [func(*args, **kwargs) for func in func_list]
    return to_ret


def fill_plot(ax, mean, std, color='b', alpha=0.3, label=None):
    ax.plot(mean, color=color, label=label)
    ax.fill_between(jnp.arange(len(mean)), mean - std, mean + std, color=color, alpha=alpha)


def array_norm_linear(array):
    """
    Linearly normalises the input by setting the min and max to 0 and 1 respectively
    """
    min_vals = array.min(axis=0)
    max_vals = array.max(axis=0)
    return (array - min_vals) / (max_vals - min_vals)

def array_norm_rank(array):
    """
    Normalise by the rank, so the output  will be uniformly distributed
    arry with the same relative order as the input
    """
    return jnp.argsort(jnp.argsort(array, axis=0), axis = 0) / (array.shape[0] - 1)

def norm_through_ind(array, norm_indicies, norm_method = None):
    """
    Applies the norming 'through' the specified indicies.
    
    The indicies should be the directions where the data will be grouped together.
    
    Example input:
    norm_through_ind(jnp.arange(6).reshape((2,3)), [1], norm_method = array_norm_rank)
    
    Output:
    Array([[0. , 0.5, 1. ],
       [0. , 0.5, 1. ]], dtype=float32)
    
    """
    if norm_method is None:
        norm_method = array_norm_linear
    array = jnp.array(array)
    arr_shape = jnp.array(array.shape)
    
    all_indcies= set(range(len(arr_shape)))
    norm_indicies = tuple(set(norm_indicies)) # weird way of sorting
    other_indicies = tuple(all_indcies - set(norm_indicies))
    
    reshaped_array = array.transpose(norm_indicies + other_indicies)
    
    reshaped_arr_shape = reshaped_array.shape
    
    reshaped_array = reshaped_array.reshape( (np.prod(reshaped_arr_shape[:len(norm_indicies)]) , -1) )
    
    # Apply norming
    reshaped_array = norm_method(reshaped_array)
    
    # Reshape back
    
    reshaped_array = reshaped_array.reshape( reshaped_arr_shape )
    
    inverse_transpose =  np.argsort( np.array(norm_indicies + other_indicies) ) 
    
    final_array = reshaped_array.transpose(inverse_transpose)
    
    return final_array
    
    