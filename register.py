import face_recognition
import sqlite3
import numpy as np
from datetime import datetime

def register_patient(image_path, patient_id, mdr_status, ward):

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print("❌ No face detected!")
        return

    embedding = encodings[0]

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Insert patient
    cursor.execute("""
    INSERT OR REPLACE INTO patients
    VALUES (?, ?, ?, ?)
    """, (patient_id, mdr_status, ward, datetime.now()))

    # Convert embedding to binary
    embedding_blob = embedding.tobytes()

    cursor.execute("""
    INSERT INTO embeddings (patient_id, embedding)
    VALUES (?, ?)
    """, (patient_id, embedding_blob))

    conn.commit()
    conn.close()

    print(f"✅ Patient {patient_id} registered successfully!")

# Example
register_patient("patient.jpg", "P001", "MDR-TB", "Ward-5")
