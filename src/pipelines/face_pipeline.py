import dlib
import numpy as np
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students


# =====================================================
# LOAD DLIB MODELS
# =====================================================

@st.cache_resource
def load_dlib_models():

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


# =====================================================
# GENERATE FACE EMBEDDINGS
# =====================================================

def get_face_embeddings(image_np):

    detector, sp, facerec = load_dlib_models()

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:

        shape = sp(image_np, face)

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
# LOAD STUDENT EMBEDDINGS
# =====================================================

@st.cache_resource
def get_trained_model():

    known_faces = []

    student_db = get_all_students()

    print("Students in DB:", len(student_db))

    if not student_db:
        return []

    for student in student_db:

        embedding = student.get("face_embedding")

        if embedding:

            known_faces.append({
                "student_id": student.get("student_id"),
                "embedding": np.array(embedding)
            })

    print("Embeddings Loaded:", len(known_faces))

    return known_faces


# =====================================================
# REFRESH CACHE AFTER REGISTRATION
# =====================================================

def train_classifier():

    st.cache_resource.clear()

    get_trained_model()

    return True


# =====================================================
# PREDICT ATTENDANCE
# =====================================================

def predict_attendance(class_image_np):

    encodings = get_face_embeddings(
        class_image_np
    )

    detected_student = {}

    known_faces = get_trained_model()

    if not known_faces:

        print("No student embeddings found")

        return (
            detected_student,
            [],
            len(encodings)
        )

    all_students = sorted(
        list(
            set(
                face["student_id"]
                for face in known_faces
            )
        )
    )

    print("Faces Detected:", len(encodings))
    print("Known Students:", all_students)

    # Start with 0.50
    # If genuine students reject ho rahe hain
    # then increase to 0.55 or 0.60
    THRESHOLD = 0.50

    for encoding in encodings:

        best_distance = float("inf")
        best_student = None

        print("\n========== FACE ==========")

        for student in known_faces:

            distance = np.linalg.norm(
                student["embedding"] - encoding
            )

            print(
                f"Student {student['student_id']} Distance: {distance}"
            )

            if distance < best_distance:

                best_distance = distance
                best_student = student["student_id"]

        print(
            f"Best Match Student: {best_student}"
        )

        print(
            f"Best Distance: {best_distance}"
        )

        if best_distance < THRESHOLD:

            detected_student[
                best_student
            ] = True

            print(
                f"Student {best_student} MATCHED"
            )

        else:

            print(
                "UNKNOWN FACE REJECTED"
            )

    return (
        detected_student,
        all_students,
        len(encodings)
    )

