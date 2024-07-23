import jax
import jax.numpy as jnp
import jax.random as jr

"""
All selection functions take as input an array of fitnesses, and output an array of probabilities of selection.
"""

def base_chance_threshold_select(fitnesses, params):

    """
    Parameters:
    - Threshold: The fitness value above which there is a 100% chance of selection.
    - Base chance: Chance of selection that all cells have irrespective of fitness.
    """

    #########################
    # 100%        ________  #
    #            |          #
    #            |          #
    #            |          #
    #            |          #
    #    ________|          #
    # 0%                    #
    #        Fitness        #
    #########################

    normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
    return jnp.clip(params['base_chance'] + (normed_fitness >= params['threshold']), 0, 1)

def sigmoid_select(fitnesses, params):

    """
    Parameters:
    - Threshold: Midway point of the upwards curve.
    - Steepness: How steep or shallow the curve is.
    """

    #########################
    # 100%           _____  #
    #              -        #
    #            -          #
    #           -           #
    #           -           #
    #          -            #
    # 0% __ -               #
    #        Fitness        #
    #########################

    normed_fitness = jnp.argsort(jnp.argsort(fitnesses))/(fitnesses.shape[0]-1)
    return (1-params['base_chance']) / (1+jnp.exp(params['steepness'] *
                                ( params['threshold'] - normed_fitness))) + params['base_chance']