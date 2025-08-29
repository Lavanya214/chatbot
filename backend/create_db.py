import sqlite3

# Connect to the database (will create if not exists)
conn = sqlite3.connect("resources.db")
cur = conn.cursor()

# Create the table
cur.execute(
    """
CREATE TABLE IF NOT EXISTS resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    document_link TEXT,
    video_link TEXT
)
"""
)

# Example: insert sample data
cur.execute(
    """
INSERT INTO resources (keyword, document_link, video_link)
VALUES
('AI', 'docs/ai_doc.pdf', 'https://youtube.com/ai_video'),
('Python', 'docs/python_doc.pdf', 'https://youtube.com/python_video')
"""
)

conn.commit()
conn.close()
print("Database and resources table created successfully!")
