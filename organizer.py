import json
import os
import shutil
from pathlib import Path

import numpy as np

from clustering import (
    cluster_faces_precise,
    compute_cluster_centroids,
    find_similar_clusters,
)
from config import (
    ANALYSIS_FILE,
    DETECTION_SIZE,
    DISTANCE_THRESHOLD,
    MERGE_SUGGESTION_THRESHOLD,
    MERGE_SUGGESTIONS_FILE,
    METHOD,
)
from face_analyzer import detect_faces, initialize_face_analyzer
from image_loader import load_images


def generate_person_description(cluster_idx, representative_file, cluster_size):
    """Generate a brief description for a person folder."""
    file_id = (
        Path(representative_file).stem[:6]
        if representative_file
        else f"{cluster_idx:04d}"
    )
    return f"Person_{cluster_idx:02d}_{file_id}"


def organize_photos(
    models_folder,
    output_folder,
    distance_threshold=DISTANCE_THRESHOLD,
    merge_threshold=MERGE_SUGGESTION_THRESHOLD,
    method=METHOD,
    detection_size=DETECTION_SIZE,
):
    """Main function to organize photos by person with high precision."""
    print("FACIAL RECOGNITION PHOTO ORGANIZER")
    print("Configuration:")
    print(f"Distance threshold: {distance_threshold} (stricter = more accurate)")
    print(f"Merge suggestion threshold: {merge_threshold}")
    print(f"Linkage method: {method}")
    print(f"Detection size: {detection_size}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Initialize face analyzer
    print("[1/6] Initializing face recognition model...")
    app = initialize_face_analyzer()
    print("      Model loaded successfully")

    # Step 2: Load all images
    print(f"[2/6] Loading images from {models_folder}...")
    images, image_files = load_images(models_folder)
    print(f"Loaded {len(images)} images")

    if len(images) == 0:
        print("No images found. Exiting.")
        return

    # Step 3: Detect faces and get embeddings
    print("[3/6] Detecting faces (high precision mode)...")
    embeddings, embedding_to_image, images_with_faces, quality_scores = detect_faces(
        images, app
    )
    print(f"Found {len(embeddings)} face(s) in {len(images_with_faces)} image(s)")
    print(f"Average detection quality: {np.mean(quality_scores):.2f}")

    if len(embeddings) == 0:
        print("No faces detected in any images. Exiting.")
        return

    # Step 4: Cluster faces to identify unique people
    print("[4/6] Clustering faces with high precision...")
    labels, embeddings_normalized = cluster_faces_precise(
        embeddings, distance_threshold
    )

    unique_labels = set(labels)
    unique_labels = {label for label in unique_labels if label >= 0}

    print(f"Initial clustering: {len(unique_labels)} unique person(s)")

    # Step 5: Analyze cluster similarities for potential merges
    print("[5/6] Analyzing cluster similarities...")
    centroids = compute_cluster_centroids(embeddings_normalized, labels)
    potential_merges = find_similar_clusters(
        centroids, labels, embeddings_normalized, merge_threshold
    )

    if potential_merges:
        print(
            f"Found {len(potential_merges)} potential merge(s) (similarity >= {merge_threshold}):"
        )
        for merge in potential_merges[:10]:
            print(
                f"Person_{merge['cluster_i']:02d} <-> Person_{merge['cluster_j']:02d}: "
                f"{merge['similarity']:.3f} similarity "
                f"({merge['count_i']}+{merge['count_j']} photos)"
            )
        if len(potential_merges) > 10:
            print(f"... and {len(potential_merges) - 10} more")
    else:
        print(f"No additional merges suggested at threshold {merge_threshold}")

    # Step 6: Group images by person and organize
    print("[6/6] Organizing photos into person folders...")

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
        merge_file = os.path.join(output_folder, MERGE_SUGGESTIONS_FILE)
        with open(merge_file, "w") as f:
            json.dump(potential_merges, f, indent=2)
        print(f"\nMerge suggestions saved to: {merge_file}")

    # Save full analysis
    analysis_file = os.path.join(output_folder, ANALYSIS_FILE)
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
    print(f"Full analysis saved to: {analysis_file}")

    return person_info, potential_merges
