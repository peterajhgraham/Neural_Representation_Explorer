import numpy as np


def simulate_spikes(n_neurons=50, n_timesteps=1000):
    """
    Simulate spike trains for neurons using Poisson firing.
    """
    firing_rates = np.random.uniform(0.1, 5, size=n_neurons)

    spikes = np.random.poisson(
        firing_rates[:, None],
        size=(n_neurons, n_timesteps)
    )

    return spikes
