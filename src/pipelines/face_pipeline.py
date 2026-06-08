import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


# =====================================================
# LOAD DLIB MODELS
# =====================================================
# This function loads:
# 1. Face Detector
# 2. Facial Landmark Predictor
# 3. Face Recognition Model
#
# Streamlit cache ensures models load only once.
# =====================================================

@st.cache_resource
def load_dlib_models():

    # Detect faces from image
    detector = dlib.get_frontal_face_detector()

    # Detect facial landmarks (eyes, nose, mouth)
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    # Generate 128-dimensional face embeddings
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


# =====================================================
# GENERATE FACE EMBEDDINGS
# =====================================================
# Input:
#   Image (NumPy Array)
#
# Output:
#   List of 128-dimensional face embeddings
#
# Face embeddings are numerical representations of faces.
# Two similar faces produce embeddings close together.
# =====================================================

def get_face_embeddings(image_np):

    detector, sp, facerec = load_dlib_models()

    # Detect faces in image
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:

        # Extract facial landmarks
        shape = sp(image_np, face)

        # Generate 128-dimensional embedding
        face_descriptor = facerec.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encodings.append(
            np.array(face_descriptor)
        )

    return encodings


# =====================================================
# TRAIN FACE RECOGNITION MODEL
# =====================================================
# Steps:
# 1. Fetch all students from database
# 2. Load stored face embeddings
# 3. Train SVM classifier
#
# Returns:
#   Trained classifier
#   X = Face embeddings
#   y = Student IDs
# =====================================================

@st.cache_resource
def get_trained_model():

    X = []   # Face embeddings
    y = []   # Student IDs

    # Fetch students from database
    student_db = get_all_students()

    print("Students in DB:", len(student_db))

    if not student_db:
        print("No students found")
        return None

    # Load stored embeddings
    for student in student_db:

        embedding = student.get('face_embedding')

        if embedding:

            X.append(
                np.array(embedding)
            )

            y.append(
                student.get('student_id')
            )

    print("Embeddings Loaded:", len(X))

    if len(X) == 0:
        print("No face embeddings available")
        return None

    # Train Support Vector Machine Classifier
    clf = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced'
    )

    try:

        clf.fit(X, y)

    except ValueError as e:

        print("Training Error:", e)
        return None

    print("Model Trained Successfully")

    return {
        'clf': clf,
        'X': X,
        'y': y
    }


# =====================================================
# RETRAIN MODEL
# =====================================================
# Called whenever a new student registers.
#
# Clears old cache and trains model again.
# =====================================================

def train_classifier():

    st.cache_resource.clear()

    model_data = get_trained_model()

    return bool(model_data)


# =====================================================
# PREDICT ATTENDANCE
# =====================================================
# Workflow:
#
# 1. Detect faces from classroom image
# 2. Generate face embeddings
# 3. Predict student using SVM
# 4. Compare embedding distance
# 5. Mark present if distance is acceptable
#
# Returns:
#   detected_student
#   all_students
#   total_faces_detected
# =====================================================

def predict_attendance(class_image_np):

    # Generate embeddings from uploaded image
    encodings = get_face_embeddings(
        class_image_np
    )

    detected_student = {}

    # Load trained model
    model_data = get_trained_model()

    if not model_data:

        print("No trained model available")

        return (
            detected_student,
            [],
            len(encodings)
        )

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    # Get unique student IDs
    all_students = sorted(
        list(set(y_train))
    )

    print("Faces Detected:", len(encodings))
    print("Known Students:", all_students)

    # Process each detected face
    for encoding in encodings:

        # Multiple students available
        if len(all_students) >= 2:

            predicted_id = int(
                clf.predict([encoding])[0]
            )

        # Only one student registered
        else:

            predicted_id = int(
                all_students[0]
            )

        print(
            "Predicted Student:",
            predicted_id
        )

        # Fetch embedding belonging to predicted student
        student_embedding = X_train[
            y_train.index(predicted_id)
        ]

        # Calculate Euclidean Distance
        # Lower distance = Higher similarity
        best_match_score = np.linalg.norm(
            student_embedding - encoding
        )

        print(
            "Distance:",
            best_match_score
        )

        # Face similarity threshold
        #
        # Lower Threshold:
        # More strict recognition
        #
        # Higher Threshold:
        # Easier recognition
        #
        # Start with 1.0 for testing.
        resemblance_threshold = 1.0

        if best_match_score <= resemblance_threshold:

            print(
                f"Student {predicted_id} Matched"
            )

            detected_student[
                predicted_id
            ] = True

        else:

            print(
                f"Student {predicted_id} Rejected"
            )

    return (
        detected_student,
        all_students,
        len(encodings)
    )