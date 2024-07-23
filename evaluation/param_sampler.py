import jax
import jax.numpy as jnp
import jax.random as jr


def param_sampler(*ranges, rng, num_samples=10):
    """
    Generate parameter samples within specified ranges.

    Parameters:
    *ranges (tuple): Variable-length argument representing the parameter ranges.
    rng (numpy.random.Generator): Random number generator.
    num_samples (int, optional): Number of samples to generate for each parameter. Default is 10.

    Returns:
    numpy.ndarray: Array of parameter samples with shape (num_parameters, num_samples).

    """
    samples = jnp.array(
        [jnp.linspace(range[0], range[1], num=num_samples) for range in ranges])
    return jr.permutation(rng, samples, axis=1, independent=True)

def TwoD_param_sampler_grid(x, y, rng, num_samples=10):
    x = jnp.linspace(x[0], x[1], num=num_samples)
    y = jnp.linspace(y[0], y[1], num=num_samples)
    X, Y = jnp.meshgrid(x,y)
    return jnp.array([X.flatten(),Y.flatten()])


def spiking_sampler_fixed_prop(base_chance_range, proportion, num_samples=10):
    """
    Generate samples of threshold and base chance values for a selection function,
    with a fixed proportion of kept samples.

    Args:
        base_chance_range (tuple): A tuple containing the range of base chance values.
        proportion (float): The desired proportion of kept samples.
        num_samples (int, optional): The number of samples to generate. Defaults to 10.

    Returns:
        numpy.ndarray: An array containing the generated threshold and base chance samples.
    """

    def spiking_integral(base_chance, proportion):
        """
        Returns the threshold required to achieve a certain keep proportion for any base_chance value.

        Args:
            base_chance (float): The base chance value.
            proportion (float): The desired proportion of kept samples.

        Returns:
            float: The threshold value.
        """
        return (1 - proportion) / (1 - base_chance)

    base_chance_range_relevant = [base_chance_range[0], min(
        proportion, base_chance_range[1])]

    base_chance_samples = jnp.linspace(
        base_chance_range_relevant[0], base_chance_range_relevant[1], num=num_samples)

    threshold_samples = spiking_integral(base_chance_samples, proportion)

    return jnp.array([threshold_samples, base_chance_samples])


def sigmoid_sampler_fixed_prop(rng, threshold_range, steepness_range, proportion, num_samples=10):
    """
    Generate samples from a sigmoid distribution with a fixed proportion.

    Args:
        rng (numpy.random.Generator): Random number generator.
        threshold_range (tuple): Range of threshold values.
        steepness_range (tuple): Range of steepness values.
        proportion (float): Desired proportion of samples.
        num_samples (int, optional): Number of samples to generate. Defaults to 10.

    Returns:
        numpy.ndarray: Array of threshold, steepness, and base chance samples.

    """
    def sigmoid_integral(threshold, steepness, proportion):
        """
        Calculate the base_chance required to achieve a certain keep proportion for any threshold/steepness combination.

        Args:
            threshold (numpy.ndarray): Array of threshold values.
            steepness (numpy.ndarray): Array of steepness values.
            proportion (float): Desired proportion of samples.

        Returns:
            numpy.ndarray: Array of base chance values.

        """
        Z = 1/steepness*(jnp.log(jnp.exp(steepness*threshold) +
                                 jnp.exp(steepness))-jnp.log(jnp.exp(steepness*threshold)+1))
        return (proportion - Z)/(1 - Z)

    threshold_range = [
        max(threshold_range[0], 1-proportion), threshold_range[1]]

    threshold_samples = jnp.linspace(
        threshold_range[0], threshold_range[1], num=num_samples)
    steepness_samples = jnp.logspace(
        jnp.log10(steepness_range[0]), jnp.log10(steepness_range[1]), num=num_samples)
    steepness_samples = jr.permutation(rng, steepness_samples)

    base_chance_samples = sigmoid_integral(
        threshold_samples, steepness_samples, proportion)

    relevant_indexes = jnp.array([base_chance_samples > 0])
    relevant_indexes = relevant_indexes*jnp.array([base_chance_samples < 1])

    return jnp.array([threshold_samples[relevant_indexes[0]],
                      steepness_samples[relevant_indexes[0]],
                      base_chance_samples[relevant_indexes[0]]])

def sigmoid_sampler_fixed_steepness(threshold_range, base_chance_range, steepness, num_samples=10):
    """
    Generate samples for a sigmoid function with a fixed steepness.

    Args:
        threshold_range (tuple): A tuple containing the minimum and maximum values for the threshold.
        base_chance_range (tuple): A tuple containing the minimum and maximum values for the base chance.
        steepness (float): The fixed steepness value for the sigmoid function.
        num_samples (int, optional): The number of samples to generate. Defaults to 10.

    Returns:
        numpy.ndarray: An array containing the generated samples for the threshold, steepness, and base chance.
    """
        
    threshold_samples = jnp.linspace(threshold_range[0], threshold_range[1], num=num_samples)
    base_chance_samples = jnp.linspace(base_chance_range[0], base_chance_range[1], num=num_samples)
    threshold_samples, base_chance_samples = jnp.meshgrid(threshold_samples, base_chance_samples)
    
    return jnp.array([threshold_samples.flatten(),jnp.array([steepness]*(num_samples**2)),base_chance_samples.flatten()])


