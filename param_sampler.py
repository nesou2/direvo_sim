import jax
import jax.numpy as jnp
import jax.random as jr


def param_sampler(*ranges, rng, num_samples=10):

    samples = jnp.array(
        [jnp.linspace(range[0], range[1], num=num_samples) for range in ranges])
    return jr.permutation(rng, samples, axis=1, independent=True)

def grid_sampler(x, y, num_samples=10):
    x = jnp.linspace(x[0], x[1], num=num_samples)
    y = jnp.linspace(y[0], y[1], num=num_samples)
    X, Y = jnp.meshgrid(x,y)
    return jnp.array([X.flatten(),Y.flatten()])


def base_chance_threshold_fixed_prop(base_chance_range, proportion, num_samples=10):

    def base_chance_threshold_integral(base_chance, proportion):
        # Returns the threshold required to achieve a certain keep proportion for any base_chance value.
        return (1-proportion)/(1-base_chance)

    base_chance_range_relevant = [base_chance_range[0], min(
        proportion, base_chance_range[1])]

    base_chance_samples = jnp.linspace(
        base_chance_range_relevant[0], base_chance_range_relevant[1], num=num_samples)

    threshold_samples = base_chance_threshold_integral(base_chance_samples, proportion)

    return jnp.array([threshold_samples, base_chance_samples])