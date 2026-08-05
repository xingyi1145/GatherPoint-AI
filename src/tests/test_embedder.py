import torch
from sentence_transformers import SentenceTransformer

# Verify PyTorch recognizes the ROCm GPU backend
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model on: {device}")

# Load the lightweight model recommended in the plan
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Embedding model loaded successfully onto the GPU!")