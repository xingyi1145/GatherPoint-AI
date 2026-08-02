import chromadb
import torch
from sentence_transformers import SentenceTransformer

# 1. Load the local model on the ROCm GPU (Offline)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedder on: {device}")
embedder = SentenceTransformer('./all-MiniLM-L6-v2', device=device)

# 2. Initialize Persistent ChromaDB (Saves to local disk)
print("Initializing local ChromaDB...")
client = chromadb.PersistentClient(path="./gatherpoint_db")
collection = client.get_or_create_collection(name="friend_profiles")

# 3. Prepare Hybrid Data (Text for Agent, Metadata for GIS Python Tools)
documents = [
    "Alice is vegan, hates driving, takes the subway, and lives at Union Station.",
    "Bob hates spicy food, loves biking, and lives at Yonge and Bloor."
]
ids = ["alice_profile", "bob_profile"]
metadatas = [
    {"home_lat": 43.6452, "home_lng": -79.3806, "default_mode": "TRANSIT", "isochrone_1h": "[]"},
    {"home_lat": 43.6704, "home_lng": -79.3868, "default_mode": "BICYCLE", "isochrone_1h": "[]"}
]

# 4. Generate embeddings locally to bypass firewall
print("Generating embeddings using AMD GPU...")
embeddings = embedder.encode(documents).tolist()

# 5. Insert into Database
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings  # Explicitly pass our local embeddings!
)

print(f"Success! Database initialized and saved to disk. Total profiles: {collection.count()}")