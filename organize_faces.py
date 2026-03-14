#!/usr/bin/env python3
"""
Facial Recognition Photo Organizer

This script analyzes all photos in the Models folder, performs facial recognition,
identifies unique individuals, and organizes photos into person-specific folders.

Uses insightface library for face detection and recognition.
"""

import os
import shutil
import numpy as np
from pathlib import Path
import cv2
import insightface
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


def initialize_face_analyzer():
    """Initialize the face analysis model."""
    print("      Loading face recognition model...")
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def load_images(folder_path):
    """Load all images from the specified folder."""
    images = []
    image_files = []
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
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
    """Detect faces in all images and return face embeddings."""
    all_embeddings = []
    embedding_to_image = []
    images_with_faces = set()
    
    for idx, image in enumerate(images):
        try:
            # Detect faces
            faces = app.get(image)
            
            if len(faces) == 0:
                continue
            
            images_with_faces.add(idx)
            
            # Get embeddings for each face
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    all_embeddings.append(face.embedding)
                    embedding_to_image.append(idx)
                    
        except Exception as e:
            print(f"Warning: Error processing image: {e}")
            continue
    
    return np.array(all_embeddings), embedding_to_image, images_with_faces


def cluster_faces_cosine(embeddings, threshold=0.5):
    """
    Cluster face embeddings using cosine similarity.
    
    For insightface embeddings:
    - Cosine similarity > 0.5 typically indicates same person
    - Distance = 1 - similarity
    """
    if len(embeddings) == 0:
        return np.array([])
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = normalize(embeddings)
    
    # Use Agglomerative Clustering with cosine distance
    # distance_threshold controls when to stop merging clusters
    # Lower threshold = more clusters (stricter), Higher = fewer clusters (more lenient)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings_normalized)
    
    return labels


def generate_person_description(cluster_idx, representative_file, cluster_size):
    """Generate a brief description for a person folder."""
    # Use the first few characters of a photo filename as identifier
    file_id = Path(representative_file).stem[:6] if representative_file else f"{cluster_idx:04d}"
    return f"Person_{cluster_idx:02d}_{file_id}"


def organize_photos(models_folder, output_folder, distance_threshold=0.5):
    """Main function to organize photos by person."""
    
    print("=" * 60)
    print("FACIAL RECOGNITION PHOTO ORGANIZER")
    print("=" * 60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Initialize face analyzer
    print(f"\n[1/5] Initializing face recognition model...")
    app = initialize_face_analyzer()
    print("      Model loaded successfully")
    
    # Step 2: Load all images
    print(f"\n[2/5] Loading images from {models_folder}...")
    images, image_files = load_images(models_folder)
    print(f"      Loaded {len(images)} images")
    
    if len(images) == 0:
        print("No images found. Exiting.")
        return
    
    # Step 3: Detect faces and get embeddings
    print(f"\n[3/5] Detecting faces...")
    embeddings, embedding_to_image, images_with_faces = detect_faces(images, app)
    print(f"      Found {len(embeddings)} face(s) in {len(images_with_faces)} image(s)")
    
    if len(embeddings) == 0:
        print("No faces detected in any images. Exiting.")
        return
    
    # Step 4: Cluster faces to identify unique people
    print(f"\n[4/5] Clustering faces to identify unique individuals...")
    print(f"      Using cosine similarity with distance threshold: {distance_threshold}")
    labels = cluster_faces_cosine(embeddings, distance_threshold)
    
    unique_labels = set(labels)
    unique_labels = {l for l in unique_labels if l >= 0}
    
    print(f"      Identified {len(unique_labels)} unique person(s)")
    
    # Step 5: Group images by person and organize
    print(f"\n[5/5] Organizing photos into person folders...")
    
    person_clusters = {}
    for idx, label in enumerate(labels):
        if label >= 0:  # Ignore noise
            if label not in person_clusters:
                person_clusters[label] = []
            person_clusters[label].append(idx)
    
    # Create folders and copy photos
    person_info = {}
    
    for person_idx, cluster_indices in person_clusters.items():
        # Get the image indices for this person
        image_indices = list(set(embedding_to_image[idx] for idx in cluster_indices))
        
        # Generate folder name
        representative_file = image_files[image_indices[0]]
        folder_name = generate_person_description(person_idx, representative_file, len(image_indices))
        
        person_folder = os.path.join(output_folder, folder_name)
        os.makedirs(person_folder, exist_ok=True)
        
        # Copy all photos for this person
        copied_files = []
        for img_idx in image_indices:
            src_path = os.path.join(models_folder, image_files[img_idx])
            dst_path = os.path.join(person_folder, image_files[img_idx])
            shutil.copy2(src_path, dst_path)
            copied_files.append(image_files[img_idx])
        
        person_info[folder_name] = {
            'person_id': person_idx,
            'photo_count': len(copied_files),
            'files': copied_files
        }
        
        print(f"      Created '{folder_name}' with {len(copied_files)} photo(s)")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(images)}")
    print(f"Images with faces: {len(images_with_faces)}")
    print(f"Total faces detected: {len(embeddings)}")
    print(f"Total unique people identified: {len(person_clusters)}")
    print(f"\nOutput folder: {output_folder}")
    print("=" * 60)
    
    # Detailed breakdown
    print("\nDetailed breakdown (people with multiple photos):")
    multi_photo_people = [(name, info) for name, info in person_info.items() if info['photo_count'] > 1]
    single_photo_people = [(name, info) for name, info in person_info.items() if info['photo_count'] == 1]
    
    if multi_photo_people:
        for folder_name, info in sorted(multi_photo_people, key=lambda x: x[1]['photo_count'], reverse=True):
            print(f"  {folder_name}: {info['photo_count']} photo(s)")
    
    print(f"\n  ... and {len(single_photo_people)} person(s) with 1 photo each")
    
    return person_info


if __name__ == "__main__":
    # Configuration
    SCRIPT_DIR = Path(__file__).parent
    MODELS_FOLDER = os.path.join(SCRIPT_DIR, "Models")
    OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Organized_People")
    
    # Distance threshold for cosine similarity clustering
    # Lower = stricter (more clusters), Higher = more lenient (fewer clusters)
    # Typical range: 0.3-0.7 for face recognition
    # 0.5 is a good starting point
    DISTANCE_THRESHOLD = 0.5
    
    # Remove existing output folder for fresh run
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    
    # Run the organization
    organize_photos(MODELS_FOLDER, OUTPUT_FOLDER, DISTANCE_THRESHOLD)
