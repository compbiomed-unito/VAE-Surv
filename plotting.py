from sklearn.manifold import TSNE
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_latent_rep(latent_train, latent_test=None):
    """
    Create a t-SNE visualization of latent features.
    
    If both latent_train and latent_test are provided, they will be combined and
    displayed as separate groups (Train vs Test). Otherwise, if only latent_train is provided,
    a single plot is created.
    
    Parameters:
        latent_train (np.ndarray): Latent features for the training (or single) dataset.
        latent_test (np.ndarray, optional): Latent features for the test dataset.
    """

    # --- Step 1: Combine or process latent features ---
    if latent_test is not None:
        # Combine train and test
        latent_combined = np.concatenate([latent_train, latent_test], axis=0)
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        latent_2d = tsne.fit_transform(latent_combined)
    
        n_train = latent_train.shape[0]
        latent_train_2d = latent_2d[:n_train, :]
        latent_test_2d  = latent_2d[n_train:, :]
    else:
        tsne = TSNE(n_components=2, perplexity=30, random_state=0)
        latent_train_2d = tsne.fit_transform(latent_train)
    
    # --- Step 2: Create the Plot ---
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    if latent_test is not None:
        # Plot training data with blue circles
        ax.scatter(
            latent_train_2d[:, 0], latent_train_2d[:, 1],
            c='royalblue', s=60, edgecolor='k', alpha=0.7,
            marker='o', label='Train'
        )
        # Plot testing data with red triangles
        ax.scatter(
            latent_test_2d[:, 0], latent_test_2d[:, 1],
            c='salmon', s=60, edgecolor='k', alpha=0.85,
            marker='^', label='Test'
        )
        ax.set_title("t-SNE Visualization of Latent Features\n(Train vs Test)",
                     fontsize=20, weight='bold', pad=20)
    else:
        # Plot single dataset with blue circles
        ax.scatter(
            latent_train_2d[:, 0], latent_train_2d[:, 1],
            c='royalblue', s=60, edgecolor='k', alpha=0.7,
            marker='o', label='Data'
        )
        ax.set_title("t-SNE Visualization of Latent Features",
                     fontsize=20, weight='bold', pad=20)
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=16)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16)
    
    # Enhance the legend
    legend = ax.legend(loc="best", fontsize=14, frameon=True, fancybox=True, framealpha=0.9)
    legend.get_frame().set_edgecolor("gray")
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()




def plot_tsne_and_survival(latent_features, cluster_labels, y_all):
    """
    Generate a t-SNE visualization of latent features colored by cluster and plot
    Kaplan–Meier survival curves for each cluster.

    Parameters:
        latent_features (np.ndarray or torch.Tensor): Latent features from the model,
            shape (num_samples, latent_dim).
        cluster_labels (np.ndarray): Cluster labels for each sample.
        y_all (structured array or dict-like): Survival data with at least the keys/fields
            "time" and "event".

    Returns:
        None
    """
    from lifelines import KaplanMeierFitter
    import matplotlib.cm as cm

    # --- Helper: Ensure data is a NumPy array ---
    def to_numpy(data):
        if hasattr(data, "cpu"):
            return data.cpu().numpy()
        return data

    # Convert latent_features to a NumPy array (if not already)
    latent_features = to_numpy(latent_features)

    # --- t-SNE Embedding ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    latent_2d = tsne.fit_transform(latent_features)  # shape: (num_samples, 2)

    # --- Create a Color Mapping for the Clusters ---
    unique_clusters = np.sort(np.unique(cluster_labels))
    num_clusters = len(unique_clusters)
    colors = sns.color_palette("tab10", n_colors=num_clusters)
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # --- Plot the t-SNE Visualization ---
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    for cluster in unique_clusters:
        indices = cluster_labels == cluster
        ax.scatter(
            latent_2d[indices, 0],
            latent_2d[indices, 1],
            c=[cluster_color_map[cluster]],  # assign the cluster-specific color
            s=70,
            edgecolor="k",
            alpha=0.8,
            label=f"Cluster {cluster}"
        )

    ax.set_title("t-SNE Visualization of Latent Representation\nColored by Cluster",
                 fontsize=20, weight="bold", pad=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=16)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16)
    legend = ax.legend(title="Cluster", fontsize=12, title_fontsize=14)
    legend.get_frame().set_edgecolor("gray")
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

    # --- Kaplan–Meier Survival Curves ---
    # Assume y_all is a structured array or dict-like object with keys "time" and "event"
    surv_df = pd.DataFrame({
        "time": y_all["time"],
        "event": y_all["event"]
    })
    surv_df["cluster"] = cluster_labels

    plt.figure(figsize=(12, 8))
    kmf = KaplanMeierFitter()

    for cluster in unique_clusters:
        cluster_df = surv_df[surv_df["cluster"] == cluster]
        kmf.fit(
            durations=cluster_df["time"],
            event_observed=cluster_df["event"],
            label=f"Cluster {cluster}"
        )
        kmf.plot_survival_function(ci_show=True, color=cluster_color_map[cluster], lw=2)

    plt.title("Kaplan–Meier Curves by Cluster", fontsize=20, weight="bold", pad=20)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Survival Probability", fontsize=16)
    legend = plt.legend(title="Cluster", fontsize=12, title_fontsize=14)
    legend.get_frame().set_edgecolor("gray")
    plt.tight_layout()
    plt.show()
