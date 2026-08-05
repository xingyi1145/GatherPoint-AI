import chromadb
import torch
from sentence_transformers import SentenceTransformer

# 1. Connect to the existing local database
client = chromadb.PersistentClient(path="./gatherpoint_db")
# Notice we use get_collection now, instead of get_or_create
collection = client.get_collection(name="friend_profiles")

# 2. Load the offline embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer('./all-MiniLM-L6-v2', device=device)

# --- HELPER 1: FOR PERSON 3 (AGENT LOOP / RAG) ---
def retrieve_user_context(query_text):
    print(f"\n[Agent Handoff] Searching DB for context matching: '{query_text}'")
    # Generate the vector for the user's prompt
    query_embedding = embedder.encode([query_text]).tolist()
    
    # Perform Semantic Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2 
    )
    
    contexts = results['documents'][0]
    print(" -> Retrieved Context to inject into System Prompt:")
    for ctx in contexts:
        print(f"    - {ctx}")
    return contexts

# --- HELPER 2: FOR PERSON 2 (GIS TOOLS) ---
def get_cached_metadata(user_id):
    print(f"\n[GIS Handoff] Fetching cached geospatial data for: {user_id}")
    # Perform exact ID lookup
    result = collection.get(ids=[user_id])
    
    if result['metadatas'] and len(result['metadatas']) > 0:
        meta = result['metadatas'][0]
        print(f" -> Retrieved Default Transit: {meta['default_mode']}")
        print(f" -> Retrieved Home Coords: {meta['home_lat']}, {meta['home_lng']}")
        return meta
    else:
        print(" -> User not found in cache.")
        return None

# --- RUN THE TESTS ---
if __name__ == "__main__":
    # Simulate the user asking the Streamlit UI a question
    retrieve_user_context("Find a good restaurant for me, Alice, and Bob.")
    
    # Simulate the GIS script bypassing the Google API
    get_cached_metadata("alice_profile")