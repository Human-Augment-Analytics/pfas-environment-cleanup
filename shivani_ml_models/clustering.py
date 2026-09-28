"""
Molecular feature clustering (KMeans + DBSCAN) with visualization.

Expects a CSV with columns like:
CID,SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight,
ExactMass,Charge,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount

Usage:
    python molecular_clustering.py --csv your_file.csv --k 4
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


# Numeric feature columns to cluster on. Adjust as needed.
FEATURE_COLS = [
    "MolecularWeight",
    "ExactMass",
    "Charge",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]

# Columns useful for labeling points in plots / output tables
ID_COLS = ["CID", "SMILES", "InChIKey", "MolecularFormula"]


def load_and_scale(csv_path, feature_cols=FEATURE_COLS):
    """Load CSV, keep only rows with complete feature data, and standardize features."""
    df = pd.read_csv(csv_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X = df[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


def run_kmeans(X_scaled, k=4, random_state=42):
    """
    Fit KMeans with k clusters.

    Returns:
        labels: cluster assignment per row
        centers: cluster centers in SCALED feature space (shape: k x n_features)
        model: fitted KMeans object
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    centers = model.cluster_centers_
    return labels, centers, model


def find_medoid_rows(X_scaled, labels, centers):
    """
    For each cluster, find the actual data point closest to the cluster center
    (i.e. the "medoid" — a real molecule, not just an abstract centroid).

    Returns a dict: {cluster_id: row_index_in_original_df}
    """
    medoid_indices = {}
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_points = X_scaled[cluster_mask]
        cluster_row_indices = np.where(cluster_mask)[0]

        dists = np.linalg.norm(cluster_points - centers[cluster_id], axis=1)
        closest_local_idx = np.argmin(dists)
        medoid_indices[cluster_id] = cluster_row_indices[closest_local_idx]

    return medoid_indices


def save_cluster_centers(centers, scaler, df, labels, medoid_indices,
                          feature_cols=FEATURE_COLS, out_path="cluster_centers.csv"):
    """
    Write cluster centers to a CSV file.

    Centers are stored in the ORIGINAL (unscaled) feature units, since that's
    what's actually interpretable. Also includes the cluster size and the
    CID/SMILES of the real molecule closest to each center (its medoid).
    """
    # Undo the StandardScaler transform to get centers back in real units
    centers_original = scaler.inverse_transform(centers)

    rows = []
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    for cluster_id in range(len(centers)):
        row = {"cluster": cluster_id, "n_points": int(cluster_sizes.get(cluster_id, 0))}
        row.update(dict(zip(feature_cols, centers_original[cluster_id])))

        medoid_idx = medoid_indices.get(cluster_id)
        if medoid_idx is not None:
            if "CID" in df.columns:
                row["medoid_CID"] = df.loc[medoid_idx, "CID"]
            if "SMILES" in df.columns:
                row["medoid_SMILES"] = df.loc[medoid_idx, "SMILES"]

        rows.append(row)

    centers_df = pd.DataFrame(rows)
    centers_df.to_csv(out_path, index=False)
    print(f"Saved cluster centers to {out_path}")
    return centers_df


def plot_clusters_2d(X_scaled, labels, centers, df, medoid_indices,
                      title="KMeans clustering (PCA projection)", save_path=None):
    """
    Project scaled features to 2D with PCA and plot:
      - all points colored by cluster
      - cluster centers marked with an X
      - the real data point closest to each center highlighted and labeled
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(centers)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab10", alpha=0.6, s=40, edgecolor="none"
    )

    # Plot the abstract cluster centers
    plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c="black", marker="X", s=200, label="Cluster center"
    )

    # Highlight the real molecule closest to each center
    for cluster_id, row_idx in medoid_indices.items():
        point_2d = X_2d[row_idx]
        plt.scatter(
            point_2d[0], point_2d[1],
            facecolor="none", edgecolor="black", linewidth=2, s=250
        )
        label = str(df.loc[row_idx, "CID"]) if "CID" in df.columns else str(row_idx)
        plt.annotate(
            label, (point_2d[0], point_2d[1]),
            textcoords="offset points", xytext=(8, 8), fontsize=9, weight="bold"
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def run_dbscan(X_scaled, eps=0.5, min_samples=5):
    """
    Fit DBSCAN. Unlike KMeans, you don't choose k — you choose eps
    (neighborhood radius) and min_samples (density threshold).
    Points labeled -1 are noise (didn't fit any cluster).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return labels, model


def suggest_dbscan_eps(X_scaled, min_samples=5, save_path=None):
    """
    Helper: k-distance elbow plot to help pick a good eps for DBSCAN.
    Look for the 'elbow' where distance sharply increases — that's a decent eps.
    """
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)

    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(7, 5))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th nearest neighbor distance")
    plt.title("k-distance plot (look for the elbow to pick eps)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_dbscan_clusters(X_scaled, labels, title="DBSCAN clustering (PCA projection)", save_path=None):
    """Same PCA-projection plot as KMeans, but for DBSCAN (noise points shown in gray)."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 7))
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        color = "lightgray" if lbl == -1 else None
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.6, s=40, c=color)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Cluster molecular features from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters for KMeans")
    parser.add_argument("--method", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--out", default=None, help="Path to save plot image (optional)")
    parser.add_argument("--centers_out", default="cluster_centers.csv",
                         help="Path to save cluster centers CSV (KMeans only)")
    args = parser.parse_args()

    df, X_scaled, scaler = load_and_scale(args.csv)

    if args.method == "kmeans":
        labels, centers, model = run_kmeans(X_scaled, k=args.k)
        medoid_indices = find_medoid_rows(X_scaled, labels, centers)

        df["cluster"] = labels
        print(df.groupby("cluster").size())

        print("\nMolecule closest to each cluster center:")
        for cluster_id, row_idx in medoid_indices.items():
            print(f"  Cluster {cluster_id}: CID={df.loc[row_idx, 'CID']}, "
                  f"SMILES={df.loc[row_idx, 'SMILES']}")

        save_cluster_centers(centers, scaler, df, labels, medoid_indices,
                              out_path=args.centers_out)

        plot_clusters_2d(X_scaled, labels, centers, df, medoid_indices, save_path=args.out)

    else:
        labels, model = run_dbscan(X_scaled, eps=args.eps, min_samples=args.min_samples)
        df["cluster"] = labels
        print(df.groupby("cluster").size())
        n_noise = int(np.sum(labels == -1))
        print(f"Noise points: {n_noise}")
        print("Note: DBSCAN has no cluster centers (no cluster_centers.csv written for this method).")
        plot_dbscan_clusters(X_scaled, labels, save_path=args.out)


if __name__ == "__main__":
    main()