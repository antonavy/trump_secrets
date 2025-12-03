#!/usr/bin/env python3

from tinydb import TinyDB, Query

# Initialize your TinyDB database
db = TinyDB('data/trump_posts_db.json')

# Get all documents from the table
all_docs = db.all()

# Check if there are any documents to delete
if all_docs:
    # The last document in the list will be the last record
    last_doc = all_docs[-1]
    last_doc_id = last_doc.doc_id

    # Remove the document using its ID
    db.remove(doc_ids=[last_doc_id])
    print(f"Removed document with ID: {last_doc_id}")
else:
    print("No documents found in the database.")

# Close the database (optional, TinyDB handles this automatically on exit)
db.close()
