import numpy as np


def compute_firing_rates(spikes, window=20):
    """
    Compute sliding window firing rates.
    """
    n_neurons, n_timesteps = spikes.shape
    features = []

    for t in range(0, n_timesteps - window):
        window_spikes = spikes[:, t:t + window]
        rates = window_spikes.mean(axis=1)
        features.append(rates)

    return np.array(features)
