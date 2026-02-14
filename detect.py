import cv2
import face_recognition
import sqlite3
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import math

# Load database
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("SELECT patient_id, embedding FROM embeddings")
rows = cursor.fetchall()

known_embeddings = []
known_ids = []

for row in rows:
    known_ids.append(row[0])
    embedding = np.frombuffer(row[1], dtype=np.float64)
    known_embeddings.append(embedding)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/hospital_video.mp4")

FRAME_TIME = 1/30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    mdr_present = False
    mdr_location = None
    mdr_id = None

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_embeddings, face_encoding)

        if True in matches:
            idx = matches.index(True)
            mdr_id = known_ids[idx]
            mdr_present = True
            mdr_location = ((left+right)//2, (top+bottom)//2)

            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.putText(frame, f"MDR: {mdr_id}", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1+x2)//2, (y1+y2)//2)

                if mdr_present:
                    distance = math.dist(center, mdr_location)
                    duration = FRAME_TIME

                    if distance < 200 and duration > 5:
                        risk = "HIGH"
                    elif distance < 200:
                        risk = "MEDIUM"
                    else:
                        risk = "LOW"

                    cursor.execute("""
                    INSERT INTO contacts
                    (patient_id, contact_person, duration, distance, risk_level, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (mdr_id, "Person", duration, distance, risk, datetime.now()))

                    conn.commit()

    cv2.imshow("MDR Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export dataset
df = pd.read_sql_query("SELECT * FROM contacts", conn)
df.to_csv("MDR_Contact_Dataset.csv", index=False)

conn.close()

print("✅ Contact dataset exported!")
