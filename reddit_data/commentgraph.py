import json
import torch
import torch.nn.functional as F
import networkx as nx
from collections import defaultdict
import itertools
import os
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Load Posts Efficiently
def load_posts(posts_file):
    """Loads posts into memory. If 10GB is too large for RAM, 
    consider a key-value store like SQLite/Redis for the texts."""
    posts = {}
    print("Loading posts into memory...")
    with open(posts_file, 'r') as f:
        for line in f:
            p = json.loads(line)
            # Use 't3_' prefix to match the link_id in comments
            post_id = p['name'] 
            text = p.get('title', '') + " " + p.get('selftext', '')
            posts[post_id] = text.strip()
    print(f"Loaded {len(posts)} posts.")
    return posts

def save_graph_to_disk(G, output_filepath, format='csv'):
    print(f"Extracting {G.number_of_edges()} edges for saving...")
    
    # Extract edges and their weights into a list of dictionaries
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append({
            'source_post': u,
            'target_post': v,
            'weight': data['weight']
        })
        
    # Convert to a Pandas DataFrame for easy exporting
    df = pd.DataFrame(edge_data)
    
    if format == 'csv':
        # Save as CSV (easy to read, but larger file size)
        df.to_csv(f"{output_filepath}.csv", index=False)
        print(f"Graph saved successfully to {output_filepath}.csv")
        
    elif format == 'parquet':
        # Save as Parquet (highly compressed, extremely fast, great for huge datasets)
        df.to_parquet(f"{output_filepath}.parquet", index=False)
        print(f"Graph saved successfully to {output_filepath}.parquet")

# 2. Generator for Chunking Comments
def comment_chunk_generator(comments_file, posts_dict, chunk_size=50000):
    """Yields batches of valid comments to prevent RAM exhaustion."""
    chunk = []
    with open(comments_file, 'r') as f:
        for line in f:
            c = json.loads(line)
            author = c.get('author')
            body = c.get('body')
            parent_post = c.get('link_id')
            
            # Filter deleted content and orphan comments
            if author in ['[deleted]', 'AutoModerator'] or body in ['[removed]', '[deleted]']:
                continue
            if parent_post not in posts_dict:
                continue
                
            chunk.append({'author': author, 'post_id': parent_post, 'text': body})
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
                
    if chunk:
        yield chunk

# 3. Vectorized Stance Calculation
def compute_stances_batched(posts, comments_file, chunk_size=50000, batch_size=2048):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Pre-compute post embeddings in bulk and keep them on the GPU
    print("Embedding all posts...")
    post_ids = list(posts.keys())
    post_texts = list(posts.values())
    
    # convert_to_tensor=True keeps data on GPU for fast similarity math
    post_embs_tensor = model.encode(post_texts, batch_size=batch_size, 
                                    convert_to_tensor=True, show_progress_bar=True)
    
    # Map post ID to its row index in the tensor
    post_idx_map = {pid: idx for idx, pid in enumerate(post_ids)}
    
    # Dictionary to hold the stance of each user on each post
    user_post_stances = defaultdict(dict)
    
    print("Processing comments in chunks...")
    for chunk_idx, comment_chunk in enumerate(comment_chunk_generator(comments_file, posts)):
        print(f"Processing comment chunk {chunk_idx + 1}...")
        
        c_texts = [c['text'] for c in comment_chunk]
        c_authors = [c['author'] for c in comment_chunk]
        c_parent_ids = [c['post_id'] for c in comment_chunk]
        
        # Embed the chunk of comments
        c_embs_tensor = model.encode(c_texts, batch_size=batch_size, convert_to_tensor=True)
        
        # Gather the corresponding parent post embeddings using indices
        parent_indices = [post_idx_map[pid] for pid in c_parent_ids]
        p_embs_tensor = post_embs_tensor[parent_indices]
        
        # Compute cosine similarity element-wise across the batch
        sim_scores = F.cosine_similarity(c_embs_tensor, p_embs_tensor)
        
        # Map similarity to discrete stances: 1, 0, -1
        # sim > 0.5 -> 1 | sim < 0.1 -> -1 | else -> 0
        stances = torch.where(sim_scores > 0.5, 1, 
                  torch.where(sim_scores < 0.1, -1, 0))
        
        # Move back to CPU to store the results
        stances = stances.cpu().numpy()
        
        for author, post_id, stance in zip(c_authors, c_parent_ids, stances):
            user_post_stances[author][post_id] = int(stance)

    return user_post_stances

# 4. Optimized Graph Construction
def build_network_optimized(user_post_stances, all_post_ids):
    print("\nBuilding post-to-post network...")
    
    # Build the final graph and explicitly add ALL nodes first (handling isolated posts)
    G = nx.Graph()
    G.add_nodes_from(all_post_ids)
    print(f"Initialized graph with {G.number_of_nodes()} total nodes.")
    
    # Accumulators for edge weights: sum of stance products, and count of shared users
    edge_weight_sum = defaultdict(int)
    edge_user_count = defaultdict(int)
    
    # Iterate by user to find shared posts (Bipartite projection)
    for user, post_stances in tqdm(user_post_stances.items(), desc="Mapping user interactions"):
        # Get all posts this specific user commented on
        interacted_posts = list(post_stances.keys())
        
        # If user commented on multiple posts, create edges between all pairs
        if len(interacted_posts) > 1:
            for p1, p2 in itertools.combinations(interacted_posts, 2):
                # Ensure consistent edge ordering
                edge = tuple(sorted((p1, p2)))
                
                stance_product = post_stances[p1] * post_stances[p2]
                
                edge_weight_sum[edge] += stance_product
                edge_user_count[edge] += 1
                
    # Add the calculated edges to the graph
    for (p1, p2), total_stance in tqdm(edge_weight_sum.items(), desc="Constructing Graph Edges"):
        count = edge_user_count[(p1, p2)]
        final_weight = total_stance / count
        G.add_edge(p1, p2, weight=final_weight)
        
    return G

# --- Execution Flow ---
if __name__ == "__main__":
    # 1. Define your absolute paths
    #POSTS_FILE = '/data/jw123/subreddits25/matched_politics_posts_2016.jsonl'
    POSTS_FILE = 'politics_cleaned2024.jsonl'
    COMMENTS_FILE = 'matched_twitter_comments2024.jsonl'
    #COMMENTS_FILE = '/data/jw123/subreddits25/matched_politics_comments_2016.jsonl'

    # 2. Run the pipeline
    print("Starting pipeline...")
    posts_dict = load_posts(POSTS_FILE)
    stances_dict = compute_stances_batched(posts_dict, COMMENTS_FILE)
    graph = build_network_optimized(stances_dict, list(posts_dict.keys()))

    # 3. Save the results to disk
    save_graph_to_disk(graph, 'network_weights', format='parquet')
    print("Pipeline complete!")