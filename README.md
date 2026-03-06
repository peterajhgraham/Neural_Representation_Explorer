# Neural Representation Explorer

Explores how high-dimensional neural spike activity forms low-dimensional structure.

## Pipeline

```
spike simulation → firing rate features → dimensionality reduction → clustering → visualization
```

1. **Simulate spikes** — Poisson spike trains for a population of neurons
2. **Compute features** — Sliding-window firing rates across the population
3. **Reduce dimensionality** — PCA and UMAP projections into 2D
4. **Cluster states** — K-Means clustering of neural population states
5. **Visualize** — Plot the neural manifold colored by cluster identity

## Getting Started

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook notebooks/explore_representations.ipynb
```

## Project Structure

```
neural-representation-explorer/
    README.md
    requirements.txt
    src/
        simulate_spikes.py
        compute_features.py
        dimensionality.py
        clustering.py
    notebooks/
        explore_representations.ipynb
```

## Example Output

Neural population states projected into 2D space using UMAP, colored by K-Means cluster labels.

## Future Work

- Apply to real neural datasets (e.g. Neuropixels recordings)
- Test different manifold learning methods (t-SNE, Isomap)
- Study temporal dynamics and state transitions
