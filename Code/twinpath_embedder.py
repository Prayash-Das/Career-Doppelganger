import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Config
csv_path = "/mmfs1/home/pdas4/Career/Dataset_Profiles_2000.csv"
index_path = "profiles_2000_faiss.index"
ids_path = "profiles_2000_ids.csv"

# Step 1: Load Data
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} records.")

# Step 2: Concatenate all columns into one string per row
# This creates a combined text field for each row using all columns.
df['combined'] = df.astype(str).agg(' '.join, axis=1)

# Step 3: Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# Step 4: Generate embeddings
texts = df['combined'].tolist()
print("🔍 Generating embeddings...")
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Step 5: Build FAISS index
# Using IndexFlatIP because we normalize embeddings (cosine similarity)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Step 6: Save aligned embeddings back into CSV
# Save the embeddings as a new column in the CSV for future reference.
df['embedding'] = [list(vec) for vec in embeddings]
df.to_csv("Profiles_2000_with_Embeddings.csv", index=False)

# Step 7: Save FAISS index and ID mapping
faiss.write_index(index, index_path)
df[['user_id']].to_csv(ids_path, index=False)

print(f"\n✅ FAISS index saved to '{index_path}'")
print(f"✅ ID mapping saved to '{ids_path}'")
print(f"✅ Aligned CSV saved to 'Profiles_2000_with_Embeddings.csv'")
