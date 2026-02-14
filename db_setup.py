import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Patient table
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    mdr_status TEXT,
    ward TEXT,
    registration_date TEXT
)
""")

# Embeddings table
cursor.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    patient_id TEXT,
    embedding BLOB,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
)
""")

# Contact log table
cursor.execute("""
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    contact_person TEXT,
    duration REAL,
    distance REAL,
    risk_level TEXT,
    date TEXT
)
""")

conn.commit()
conn.close()

print("✅ Database and tables created successfully!")
