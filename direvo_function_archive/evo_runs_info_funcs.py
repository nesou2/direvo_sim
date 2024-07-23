import jax
import jax.numpy as jnp
import jax.random as jr


default_extra_info = {
    "fitness": lambda *args, **kwargs : kwargs["fitnesses"],
    "pop": lambda *args, **kwargs : kwargs["pop"],
    "indicies": lambda *args, **kwargs : kwargs["indicies"],
}