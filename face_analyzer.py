import numpy as np
from insightface.app import FaceAnalysis

from config import DETECTION_SIZE


def initialize_face_analyzer():
    """Initialize the face analysis model with high-precision settings."""
    print("      Loading face recognition model (high precision mode)...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    # Use larger detection size for better quality embeddings
    app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    return app


def detect_faces(images, app):
    """
    Detect faces in all images and return face embeddings with quality scores.
    Returns:
        embeddings: np.array of shape (n_faces, embedding_dim)
        embedding_to_image: list of image indices corresponding to each embedding
        images_with_faces: set of image indices that contain at least one face
        quality_scores: np.array of quality scores for each face
    """
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
