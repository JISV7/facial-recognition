#!/usr/bin/env python3
"""
Facial Recognition Photo Organizer - High Precision Version

This script analyzes all photos in the Models folder, performs facial recognition,
identifies unique individuals, and organizes photos into person-specific folders.

Uses insightface library for face detection and recognition with high-precision settings.
Optimized for accuracy over speed.

Key improvements:
- Stricter distance threshold (0.35 vs 0.5)
- Complete linkage clustering (more conservative)
- Larger detection size for better embeddings
- Post-clustering verification with centroid comparison
- Pairwise similarity analysis for merge suggestions
"""

"This code is superseded by its modular aproach "
import json
import os
import shutil
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def initialize_face_analyzer():
    """Initialize the face analysis model with high-precision settings."""
    print("      Loading face recognition model (high precision mode)...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    # Use larger detection size for better quality embeddings
    # This takes more memory and time but produces more accurate results
    app.prepare(ctx_id=0, det_size=(1920, 1920))
    return app


def load_images(folder_path):
    """Load all images from the specified folder."""
    images = []
    image_files = []

    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for file_name in sorted(os.listdir(folder_path)):
        ext = Path(file_name).suffix.lower()
        if ext in supported_extensions:
            file_path = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_files.append(file_name)
                else:
                    print(f"Warning: Could not read {file_name}")
            except Exception as e:
                print(f"Warning: Error loading {file_name}: {e}")

    return images, image_files


def detect_faces(images, app):
    """Detect faces in all images and return face embeddings with quality scores."""
    all_embeddings = []
    embedding_to_image = []
    embedding_quality = []
    images_with_faces = set()

    for idx, image in enumerate(images):
        try:
            # Detect faces with high precision
            faces = app.get(image)

            if len(faces) == 0:
                continue

            images_with_faces.add(idx)

            # Get embeddings for each face
            for face in faces:
                if hasattr(face, "embedding") and face.embedding is not None:
                    all_embeddings.append(face.embedding)
                    embedding_to_image.append(idx)

                    # Calculate quality score based on detection confidence
                    # Higher bbox score = better quality detection
                    if hasattr(face, "bbox") and face.bbox is not None:
                        bbox_area = (face.bbox[2] - face.bbox[0]) * (
                            face.bbox[3] - face.bbox[1]
                        )
                        quality = min(1.0, bbox_area / 10000)  # Normalize quality
                    else:
                        quality = 0.5
                    embedding_quality.append(quality)

        except Exception as e:
            print(f"Warning: Error processing image: {e}")
            continue

    return (
        np.array(all_embeddings),
        embedding_to_image,
        images_with_faces,
        np.array(embedding_quality),
    )


def compute_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix for all embeddings."""
    # Normalize embeddings
    embeddings_normalized = normalize(embeddings)
    # Compute cosine distance matrix
    distance_matrix = pairwise_distances(embeddings_normalized, metric="cosine")
    # Convert to similarity (1 - distance)
    similarity_matrix = 1 - distance_matrix
    return similarity_matrix


def cluster_faces_precise(embeddings, distance_threshold=0.35):
    """
    Cluster face embeddings using high-precision settings.

    Key parameters:
    - distance_threshold: 0.35 is stricter than default 0.5
      - Lower = more clusters (stricter, fewer false merges)
      - Higher = fewer clusters (more lenient, more false merges)
    - linkage: 'complete' ensures all faces in cluster are similar
    """
    if len(embeddings) == 0:
        return np.array([]), None

    # Normalize embeddings for cosine similarity
    embeddings_normalized = normalize(embeddings)

    # Use Agglomerative Clustering with complete linkage
    # Complete linkage: maximum distance between clusters
    # This is more conservative than average linkage
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="complete",  # More conservative than 'average'
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
            # Centroid is the mean of all embeddings in cluster
            centroid = np.mean(cluster_embeddings, axis=0)
            # Re-normalize centroid
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
            if i < j:  # Only compare each pair once
                centroid_i = centroids[cluster_i]
                centroid_j = centroids[cluster_j]

                # Cosine similarity between centroids
                similarity = float(np.dot(centroid_i, centroid_j))

                if similarity >= threshold:
                    # Count photos in each cluster
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

    # Sort by similarity (highest first)
    potential_merges.sort(key=lambda x: x["similarity"], reverse=True)

    return potential_merges


def generate_person_description(cluster_idx, representative_file, cluster_size):
    """Generate a brief description for a person folder."""
    file_id = (
        Path(representative_file).stem[:6]
        if representative_file
        else f"{cluster_idx:04d}"
    )
    return f"Person_{cluster_idx:02d}_{file_id}"


def organize_photos(
    models_folder, output_folder, distance_threshold=0.35, merge_threshold=0.75
):
    """Main function to organize photos by person with high precision."""

    print("FACIAL RECOGNITION PHOTO ORGANIZER")
    print(f"\nConfiguration:")
    print(f"  Distance threshold: {distance_threshold} (stricter = more accurate)")
    print(f"  Merge suggestion threshold: {merge_threshold}")
    print(f"  Linkage method: complete (conservative)")
    print(f"  Detection size: 1920x1920 (high quality)")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Initialize face analyzer
    print(f"\n[1/6] Initializing face recognition model...")
    app = initialize_face_analyzer()
    print("      Model loaded successfully")

    # Step 2: Load all images
    print(f"\n[2/6] Loading images from {models_folder}...")
    images, image_files = load_images(models_folder)
    print(f"      Loaded {len(images)} images")

    if len(images) == 0:
        print("No images found. Exiting.")
        return

    # Step 3: Detect faces and get embeddings
    print(f"\n[3/6] Detecting faces (high precision mode)...")
    embeddings, embedding_to_image, images_with_faces, quality_scores = detect_faces(
        images, app
    )
    print(f"      Found {len(embeddings)} face(s) in {len(images_with_faces)} image(s)")
    print(f"      Average detection quality: {np.mean(quality_scores):.2f}")

    if len(embeddings) == 0:
        print("No faces detected in any images. Exiting.")
        return

    # Step 4: Cluster faces to identify unique people
    print(f"\n[4/6] Clustering faces with high precision...")
    labels, embeddings_normalized = cluster_faces_precise(
        embeddings, distance_threshold
    )

    unique_labels = set(labels)
    unique_labels = {l for l in unique_labels if l >= 0}

    print(f"      Initial clustering: {len(unique_labels)} unique person(s)")

    # Step 5: Analyze cluster similarities for potential merges
    print(f"\n[5/6] Analyzing cluster similarities...")
    centroids = compute_cluster_centroids(embeddings_normalized, labels)
    potential_merges = find_similar_clusters(
        centroids, labels, embeddings_normalized, merge_threshold
    )

    if potential_merges:
        print(
            f"      Found {len(potential_merges)} potential merge(s) (similarity >= {merge_threshold}):"
        )
        for merge in potential_merges[:10]:  # Show top 10
            print(
                f"        Person_{merge['cluster_i']:02d} <-> Person_{merge['cluster_j']:02d}: "
                f"{merge['similarity']:.3f} similarity "
                f"({merge['count_i']}+{merge['count_j']} photos)"
            )
        if len(potential_merges) > 10:
            print(f"        ... and {len(potential_merges) - 10} more")
    else:
        print(f"      No additional merges suggested at threshold {merge_threshold}")

    # Step 6: Group images by person and organize
    print(f"\n[6/6] Organizing photos into person folders...")

    person_clusters = {}
    for idx, label in enumerate(labels):
        if label >= 0:
            if label not in person_clusters:
                person_clusters[label] = []
            person_clusters[label].append(idx)

    # Create folders and copy photos
    person_info = {}

    for person_idx, cluster_indices in person_clusters.items():
        image_indices = list(set(embedding_to_image[idx] for idx in cluster_indices))

        representative_file = image_files[image_indices[0]]
        folder_name = generate_person_description(
            person_idx, representative_file, len(image_indices)
        )

        person_folder = os.path.join(output_folder, folder_name)
        os.makedirs(person_folder, exist_ok=True)

        copied_files = []
        for img_idx in image_indices:
            src_path = os.path.join(models_folder, image_files[img_idx])
            dst_path = os.path.join(person_folder, image_files[img_idx])
            shutil.copy2(src_path, dst_path)
            copied_files.append(image_files[img_idx])

        person_info[folder_name] = {
            "person_id": person_idx,
            "photo_count": len(copied_files),
            "files": copied_files,
            "cluster_id": person_idx,
        }

        print(f"      Created '{folder_name}' with {len(copied_files)} photo(s)")

    # Summary
    print("SUMMARY")
    print(f"Total images processed: {len(images)}")
    print(f"Images with faces: {len(images_with_faces)}")
    print(f"Total faces detected: {len(embeddings)}")
    print(f"Total unique people identified: {len(person_clusters)}")
    print(f"\nOutput folder: {output_folder}")

    # Detailed breakdown
    print("\nDetailed breakdown (people with multiple photos):")
    multi_photo_people = [
        (name, info) for name, info in person_info.items() if info["photo_count"] > 1
    ]
    single_photo_people = [
        (name, info) for name, info in person_info.items() if info["photo_count"] == 1
    ]

    if multi_photo_people:
        for folder_name, info in sorted(
            multi_photo_people, key=lambda x: x[1]["photo_count"], reverse=True
        ):
            print(f"  {folder_name}: {info['photo_count']} photo(s)")

    print(f"\n  ... and {len(single_photo_people)} person(s) with 1 photo each")

    # Save merge suggestions to file
    if potential_merges:
        merge_file = os.path.join(output_folder, "merge_suggestions.json")
        with open(merge_file, "w") as f:
            json.dump(potential_merges, f, indent=2)
        print(f"\n  Merge suggestions saved to: {merge_file}")

    # Save full analysis
    analysis_file = os.path.join(output_folder, "analysis.json")
    # Convert numpy types to Python native types for JSON serialization
    analysis_data = {
        "total_images": int(len(images)),
        "total_faces": int(len(embeddings)),
        "unique_people": int(len(person_clusters)),
        "person_info": {},
        "potential_merges": potential_merges,
    }
    for k, v in person_info.items():
        analysis_data["person_info"][k] = {
            "person_id": int(v["person_id"]),
            "photo_count": int(v["photo_count"]),
            "cluster_id": int(v["cluster_id"]),
        }
    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  Full analysis saved to: {analysis_file}")

    return person_info, potential_merges


if __name__ == "__main__":
    # Configuration
    SCRIPT_DIR = Path(__file__).parent
    MODELS_FOLDER = os.path.join(SCRIPT_DIR, "Models")
    OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Organized_People")

    # HIGH PRECISION SETTINGS
    # Distance threshold for clustering (lower = stricter)
    # 0.65-0.70 provides better grouping for faces with variations (uniform, angles)
    DISTANCE_THRESHOLD = 0.8

    # Merge suggestion threshold (cosine similarity)
    # Clusters with similarity above this will be suggested for manual review
    # 0.70 means 70% similar embeddings should be reviewed
    MERGE_SUGGESTION_THRESHOLD = 0.80

    # Remove existing output folder for fresh run
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    # Run the organization
    organize_photos(
        MODELS_FOLDER, OUTPUT_FOLDER, DISTANCE_THRESHOLD, MERGE_SUGGESTION_THRESHOLD
    )
