import face_recognition
import pickle
import os

database_path = "database/embeddings.pkl"

def register_patient(image_path, patient_id, mdr_status):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("❌ No face detected!")
        return

    embedding = encodings[0]

    data = {}

    if os.path.exists(database_path):
        with open(database_path, "rb") as f:
            data = pickle.load(f)

    data[patient_id] = {
        "embedding": embedding,
        "mdr_status": mdr_status
    }

    with open(database_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Patient {patient_id} registered successfully!")

# Example usage
register_patient("patient.jpg", "P001", "MDR-TB")
