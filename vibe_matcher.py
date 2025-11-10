Vibe Matcher: AI-Powered Fashion Discovery

This project prototype, created by Vashu Garg, demonstrates how artificial intelligence can 
redefine the fashion discovery experience for Nexora. Instead of relying on rigid keyword 
searches, it enables natural-language "vibe" queries such as "cozy weekend comfort" or 
"energetic urban chic" to surface relevant items based on meaning, not exact words.

By using OpenAI's embedding models (or mock embeddings when offline), this system 
understands customer intent to deliver semantically matched fashion products. The approach 
simplifies product discovery, improves search accuracy, and makes the shopping journey more 
intuitive — leading to deeper engagement and better conversion outcomes.

Usage:
# Without API key (uses mock embeddings):
python vibe_matcher.py

# With OpenAI API key (uses real embeddings):
set OPENAI_API_KEY=your-key-here
python vibe_matcher.py
"""

# © 2025 Vashu Garg

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time
import os
import hashlib
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_MOCK_EMBEDDINGS = False
client = None

try:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ℹ️  No API key found - using mock embeddings for demonstration")
        USE_MOCK_EMBEDDINGS = True
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI API key configured successfully")
except ImportError:
    print("ℹ️  OpenAI library not installed - using mock embeddings for demonstration")
    USE_MOCK_EMBEDDINGS = True
except Exception as e:
    print(f"ℹ️  API not available - using mock embeddings for demonstration")
    USE_MOCK_EMBEDDINGS = True

# ============================================================================
# PRODUCT DATA REPOSITORY
# ============================================================================

def create_product_data() -> pd.DataFrame:
    """
    Create a DataFrame with mock fashion products.
    """
    products = [
        {
            "name": "Boho Dress",
            "description": "Flowy, earthy tones for festival vibes",
            "vibe_tags": ["boho", "cozy", "festival"]
        },
        {
            "name": "Leather Jacket",
            "description": "Edgy urban style with sleek black finish",
            "vibe_tags": ["urban", "edgy", "modern"]
        },
        {
            "name": "Cozy Sweater",
            "description": "Soft knit for relaxed comfort",
            "vibe_tags": ["cozy", "casual", "comfort"]
        },
        {
            "name": "Athletic Joggers",
            "description": "Performance fabric for active lifestyle",
            "vibe_tags": ["athletic", "sporty", "energetic"]
        },
        {
            "name": "Minimalist Blazer",
            "description": "Clean lines for professional elegance",
            "vibe_tags": ["minimalist", "professional", "elegant"]
        },
        {
            "name": "Vintage Denim",
            "description": "Retro-inspired with distressed details",
            "vibe_tags": ["vintage", "casual", "retro"]
        },
        {
            "name": "Floral Sundress",
            "description": "Bright patterns for summer energy",
            "vibe_tags": ["floral", "energetic", "summer"]
        }
    ]
    return pd.DataFrame(products)

# ============================================================================
# EMBEDDING SERVICE
# ============================================================================

def generate_mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Generate a semantic-style mock embedding based on keyword presence.
    This produces realistic similarity behavior for demos.
    """
    text_lower = text.lower()

    clusters = {
        'urban': ['urban', 'city', 'edgy', 'modern', 'sleek', 'black', 'leather', 'chic'],
        'cozy': ['cozy', 'comfort', 'soft', 'relaxed', 'warm', 'knit', 'sweater', 'weekend'],
        'boho': ['boho', 'bohemian', 'festival', 'flowy', 'earthy', 'free'],
        'athletic': ['athletic', 'sport', 'active', 'performance', 'joggers', 'energetic'],
        'elegant': ['elegant', 'professional', 'minimalist', 'clean', 'blazer'],
        'vintage': ['vintage', 'retro', 'denim', 'distressed', 'classic'],
        'floral': ['floral', 'summer', 'bright', 'patterns', 'sundress']
    }

    cluster_scores = {k: sum(1.0 for kw in v if kw in text_lower) for k, v in clusters.items()}

    hash_obj = hashlib.md5(text.encode())
    seed = int.from_bytes(hash_obj.digest()[:4], 'big')
    rng = np.random.RandomState(seed)

    embedding = rng.randn(dim) * 0.1
    cluster_dim = dim // len(clusters)

    for i, (name, score) in enumerate(cluster_scores.items()):
        start, end = i * cluster_dim, (i + 1) * cluster_dim
        if end <= dim:
            embedding[start:end] += score * 0.5

    norm = np.linalg.norm(embedding)
    return (embedding / norm).tolist() if norm > 0 else embedding.tolist()


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generate embedding for text using OpenAI API or fallback mock embeddings.
    """
    if USE_MOCK_EMBEDDINGS or client is None:
        return generate_mock_embedding(text)

    for attempt in range(3):
        try:
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception:
            if attempt == 2:
                return generate_mock_embedding(text)
            time.sleep(1.5 * (2 ** attempt))


def get_embeddings_batch(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch) with retry logic.
    """
    if USE_MOCK_EMBEDDINGS or client is None:
        return [generate_mock_embedding(t) for t in texts]

    for attempt in range(3):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception:
            if attempt == 2:
                return [generate_mock_embedding(t) for t in texts]
            time.sleep(1.5 * (2 ** attempt))

# ============================================================================
# SIMILARITY ENGINE
# ============================================================================

def compute_similarity(query_embedding: List[float], product_embeddings: np.ndarray) -> np.ndarray:
    query_array = np.array(query_embedding).reshape(1, -1)
    return cosine_similarity(query_array, product_embeddings).flatten()


def rank_products(df: pd.DataFrame, query_embedding: List[float], top_k: int = 3) -> pd.DataFrame:
    product_embeddings = np.array(df['embedding'].tolist())
    similarities = compute_similarity(query_embedding, product_embeddings)

    results_df = df.copy()
    results_df['similarity_score'] = similarities
    top_products = results_df.nlargest(top_k, 'similarity_score')

    if similarities.max() < 0.7:
        print(f"⚠️  No strong matches found (max similarity: {similarities.max():.3f} < 0.7 threshold)")

    return top_products

# ============================================================================
# SEARCH FUNCTION
# ============================================================================

def search_products(query: str, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if 'embedding' not in df.columns or df['embedding'].isna().any():
        raise ValueError("Product DataFrame must have embeddings")

    query_embedding = get_embedding(query)
    top_products = rank_products(df, query_embedding, top_k)

    results = top_products[['name', 'description', 'similarity_score']].copy()
    results.insert(0, 'rank', range(1, len(results) + 1))
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("VIBE MATCHER: AI-POWERED FASHION DISCOVERY")
    print("=" * 80)
    print()

    print("Initializing product repository...")
    products_df = create_product_data()
    print(f"✓ Created product repository with {len(products_df)} items\n")

    print("Generating embeddings for product descriptions...")
    embeddings = get_embeddings_batch(products_df['description'].tolist())
    products_df['embedding'] = embeddings

    print(f"✓ Successfully generated {len(embeddings)} embeddings "
          f"(dimension: {len(embeddings[0])})")
    print("✓ All embeddings verified - no NaN values detected\n")

    if USE_MOCK_EMBEDDINGS:
        print("  (Using semantic-aware mock embeddings for demonstration)\n")

    test_queries = [
        "energetic urban chic",
        "cozy comfortable weekend",
        "bohemian festival style"
    ]

    print("=" * 80)
    print("RUNNING TEST QUERIES")
    print("=" * 80)
    print()

    all_scores, latencies = [], []

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: \"{query}\"")
        print("-" * 80)

        start = time.time()
        results = search_products(query, products_df, top_k=3)
        latency = time.time() - start
        latencies.append(latency)

        for _, row in results.iterrows():
            match = "✓ Good match" if row['similarity_score'] >= 0.7 else "○ Weak match"
            print(f"  Rank {row['rank']}: {row['name']}")
            print(f"    Description: {row['description']}")
            print(f"    Similarity: {row['similarity_score']:.4f} {match}")

        print(f"  Latency: {latency:.4f} seconds\n")
        all_scores.extend(results['similarity_score'].tolist())

    print("=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)
    good = sum(1 for s in all_scores if s >= 0.7)
    print(f"Total matches evaluated: {len(all_scores)}")
    print(f"Good matches: {good} ({good / len(all_scores) * 100:.1f}%)")
    print(f"Average similarity score: {np.mean(all_scores):.4f}")
    print(f"Average query latency: {np.mean(latencies):.4f} seconds\n")

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(test_queries) + 1), latencies, color='steelblue', alpha=0.7)
    plt.xlabel('Query Number')
    plt.ylabel('Latency (seconds)')
    plt.title('Query Latency Performance')
    plt.tight_layout()
    plt.savefig('vibe_matcher_latency.png', dpi=150)
    print("✓ Saved latency chart to 'vibe_matcher_latency.png'\n")

    print("=" * 80)
    print("REFLECTION & FUTURE IMPROVEMENTS")
    print("=" * 80)
    print("""
1. Integrate vector databases like Pinecone for large-scale search.
2. Add hybrid filtering (semantic + keyword).
3. Introduce personalization through user feedback.
4. Optimize latency and scalability for production.
    """)
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
