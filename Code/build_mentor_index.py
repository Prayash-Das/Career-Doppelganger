# build_mentor_index.py
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MENTOR_INPUT = "Mentor_Profiles_2000.csv"
MENTOR_OUTPUT = "Mentor_Profiles_2000_with_Embeddings.csv"
INDEX_OUTPUT = "mentor_2000_faiss.index"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

print("🔍 Generating mentor embeddings...")

df = pd.read_csv(MENTOR_INPUT)
embedder = SentenceTransformer(EMBED_MODEL)

texts = df.apply(lambda row: f"{row['name']} is a {row['current_title']} at {row['company']} in {row['industry']}, skilled in {row['skills']}, career path: {row['career_path']}", axis=1)
embeddings = embedder.encode(texts, normalize_embeddings=True).astype("float32")
df["embedding"] = embeddings.tolist()

faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

df.to_csv(MENTOR_OUTPUT, index=False)
faiss.write_index(faiss_index, INDEX_OUTPUT)
print("✅ Mentor index and metadata saved.")
