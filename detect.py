import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import math

# Load MDR Database
with open("database/embeddings.pkl", "rb") as f:
    database = pickle.load(f)

known_embeddings = []
known_ids = []

for pid in database:
    known_embeddings.append(database[pid]["embedding"])
    known_ids.append(pid)

# Load YOLO model
model = YOLO("yolov8n.pt")

video_path = "videos/hospital_video.mp4"
cap = cv2.VideoCapture(video_path)

contact_log = []

FRAME_TIME = 1/30  # assuming 30 fps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    mdr_present = False
    mdr_location = None
    mdr_id = None

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_embeddings, face_encoding)
        face_distances = face_recognition.face_distance(known_embeddings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            mdr_id = known_ids[best_match_index]
            mdr_present = True
            mdr_location = ((left+right)//2, (top+bottom)//2)

            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.putText(frame, f"MDR: {mdr_id}", (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Person detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1+x2)//2, (y1+y2)//2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if mdr_present:
                    distance = math.dist(center, mdr_location)

                    if distance < 200:
                        duration = FRAME_TIME

                        if duration > 10:
                            risk = "HIGH"
                        elif duration > 5:
                            risk = "MEDIUM"
                        else:
                            risk = "LOW"

                        contact_log.append([
                            mdr_id,
                            "Person",
                            round(duration,2),
                            round(distance,2),
                            risk,
                            datetime.now()
                        ])

    cv2.imshow("MDR Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export CSV
df = pd.DataFrame(contact_log, columns=[
    "Patient_ID", "Contact_Person",
    "Duration(sec)", "Distance(px)",
    "Risk_Level", "Date"
])

df.to_csv("MDR_Contact_Dataset.csv", index=False)

print("✅ Dataset exported successfully!")
