def kmeans_clustering(latent_features, best_k):
    """
    Perform KMeans clustering on latent_features using a preselected number of clusters (best_k)
    and plot the silhouette diagram.

    Parameters:
        latent_features (np.ndarray or torch.Tensor): Array of latent features with shape (num_samples, latent_dim).
        best_k (int): The preselected number of clusters.

    Returns:
        cluster_labels (np.ndarray): Cluster labels assigned by KMeans.
    """
    import numpy as np
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Convert to a NumPy array if latent_features is a torch.Tensor
    if isinstance(latent_features, torch.Tensor):
        latent_features = latent_features.cpu().numpy()

    # Initialize and fit KMeans with the given number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=0, max_iter=2000, n_init='auto')
    cluster_labels = kmeans.fit_predict(latent_features)

    # Compute the overall silhouette score and per-sample silhouette values
    silhouette_avg = silhouette_score(latent_features, cluster_labels, metric='euclidean')
    sample_silhouette_values = silhouette_samples(latent_features, cluster_labels, metric='euclidean')

    print(f"Average Silhouette Score for k = {best_k}: {silhouette_avg:.2f}")

    # Plot the silhouette diagram for the selected number of clusters
    fig, ax1 = plt.subplots(figsize=(10, 7))
    y_lower = 10  # Starting y-axis coordinate for the first cluster

    for i in range(best_k):
        # Collect silhouette scores for samples in cluster i and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        # Choose a color for the cluster
        color = cm.nipy_spectral(float(i) / best_k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=12)
        y_lower = y_upper + 10  # Add spacing between clusters

    # Draw a vertical line for the average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f"Average Silhouette = {silhouette_avg:.2f}")
    ax1.set_title(f"Silhouette Plot for k = {best_k}", fontsize=16)
    ax1.set_xlabel("Silhouette Coefficient Values", fontsize=14)
    ax1.set_ylabel("Cluster Label", fontsize=14)
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.legend(fontsize=12)
    plt.show()

    return cluster_labels
