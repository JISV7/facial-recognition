import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from config import METHOD


def compute_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix for all embeddings."""
    embeddings_normalized = normalize(embeddings)
    distance_matrix = pairwise_distances(embeddings_normalized, metric="cosine")
    similarity_matrix = 1 - distance_matrix
    return similarity_matrix


def cluster_faces_precise(embeddings, distance_threshold=0.35):
    """
    Cluster face embeddings using high-precision settings.
    Returns:
        labels: array of cluster labels for each embedding
        embeddings_normalized: normalized embeddings (for later use)
    """
    if len(embeddings) == 0:
        return np.array([]), None

    embeddings_normalized = normalize(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage=METHOD,
    )
    labels = clustering.fit_predict(embeddings_normalized)

    return labels, embeddings_normalized


def compute_cluster_centroids(embeddings_normalized, labels):
    """Compute centroid for each cluster."""
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        if label >= 0:
            cluster_mask = labels == label
            cluster_embeddings = embeddings_normalized[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[label] = centroid

    return centroids


def find_similar_clusters(centroids, labels, embeddings_normalized, threshold=0.75):
    """
    Find clusters that might be the same person based on centroid similarity.
    Returns list of potential merges with similarity scores.
    """
    cluster_ids = list(centroids.keys())
    potential_merges = []

    for i, cluster_i in enumerate(cluster_ids):
        for j, cluster_j in enumerate(cluster_ids):
            if i < j:
                centroid_i = centroids[cluster_i]
                centroid_j = centroids[cluster_j]

                similarity = float(np.dot(centroid_i, centroid_j))

                if similarity >= threshold:
                    count_i = int(np.sum(labels == cluster_i))
                    count_j = int(np.sum(labels == cluster_j))

                    potential_merges.append(
                        {
                            "cluster_i": int(cluster_i),
                            "cluster_j": int(cluster_j),
                            "similarity": similarity,
                            "count_i": count_i,
                            "count_j": count_j,
                        }
                    )

    potential_merges.sort(key=lambda x: x["similarity"], reverse=True)
    return potential_merges
