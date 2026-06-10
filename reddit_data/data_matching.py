from collections import defaultdict
import random
import json
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration & Paths ---
ISOT_FILE = 'ISOT.csv' # Assuming you combined True/Fake as discussed
POSTS_FILE = '/data/subreddits25/politics_submissions_2016.jsonl'
COMMENTS_FILE = '/data/subreddits25/politics_comments_2016.jsonl'

MATCHED_POSTS_OUTPUT = '/data/subreddits25/matched_politics_posts_2016.jsonl'
MATCHED_COMMENTS_OUTPUT = '/data/subreddits25/matched_politics_comments_2016.jsonl'

# Filtering Thresholds
SCORE_THRESHOLD = 10        # Minimum upvotes to be considered "highly engaging"
COMMENT_THRESHOLD = 5       # Minimum comments to ensure there is graph structure
SIMILARITY_THRESHOLD = 0.8  # Paper's strict similarity threshold
TIME_WINDOW_SEC = 3 * 24 * 60 * 60 # 3 days in seconds

def load_and_prep_isot(isot_path):
    print("Loading ISOT dataset...")
    df = pd.read_csv(isot_path)
    
    # 1. Clean the messy strings (Strips hidden spaces and newlines)
    df['date'] = df['date'].astype(str).str.strip()
    
    # 2. Standardize publication dates using multiple date formats 
    # Using format='mixed' forces Pandas to dynamically switch parsing logic row-by-row
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    
    # Drop the genuinely corrupted rows (e.g., the ones that were just URLs)
    df = df.dropna(subset=['date', 'title'])
    
    # 3. Print the class balance BEFORE returning so you can actually see it!
    print("\nClass balance of retained articles:")
    print(df['label'].value_counts())
    
    # 4. Align to reference date (Unix timestamp) for time-window matching
    df['unix_time'] = df['date'].astype('int64') // 10**9 
    
    titles = df['title'].tolist()
    timestamps = torch.tensor(df['unix_time'].tolist(), dtype=torch.long)
    labels = df['label'].tolist() 
    
    print(f"\nLoaded {len(titles)} valid ISOT articles.")
    return titles, timestamps, labels

def filter_reddit_posts(posts_path):
    print("Filtering Reddit posts by engagement...")
    filtered_posts = []
    
    with open(posts_path, 'r') as f:
        for line in tqdm(f, desc="Scanning Submissions"):
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue # Skip corrupted JSON lines entirely
            
            # 1. Defensively extract or construct the post name
            post_name = p.get('name')
            if not post_name:
                post_id = p.get('id')
                if post_id:
                    post_name = f"t3_{post_id}"
                else:
                    continue # If there is no ID at all, we can't use it in our graph
            
            # 2. Extract engagement metrics (default to 0 if missing)
            score = p.get('score', 0)
            num_comments = p.get('num_comments', 0)
            title = p.get('title', '')
            
            # 3. Apply engagement filters
            if score >= SCORE_THRESHOLD and num_comments >= COMMENT_THRESHOLD and title:
                filtered_posts.append({
                    'id': post_name, 
                    'title': title,
                    'created_utc': int(p.get('created_utc', 0)),
                    'original_data': p # Keep everything for saving later
                })
                
    print(f"Retained {len(filtered_posts)} highly engaging posts.")
    return filtered_posts

def match_datasets(reddit_posts, isot_titles, isot_timestamps, isot_labels, batch_size=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing SentenceTransformer on {device}...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # 1. Embed all ISOT titles at once
    print("Embedding ISOT news titles...")
    isot_embs = model.encode(isot_titles, convert_to_tensor=True, show_progress_bar=True)
    
    # CRITICAL FIX: Normalize ISOT embeddings here so we can use dot product later
    isot_embs = F.normalize(isot_embs, p=2, dim=1) 
    isot_timestamps = isot_timestamps.to(device)
    
    matched_results = []
    
    # 2. Process Reddit posts in batches to avoid OOM errors
    print("Semantic and Temporal Matching...")
    for i in tqdm(range(0, len(reddit_posts), batch_size), desc="Matching Batches"):
        batch = reddit_posts[i:i + batch_size]
        batch_titles = [p['title'] for p in batch]
        batch_times = torch.tensor([p['created_utc'] for p in batch], device=device)
        
        # Embed the Reddit chunk
        reddit_embs = model.encode(batch_titles, convert_to_tensor=True, show_progress_bar=False)
        
        # CRITICAL FIX: Normalize Reddit embeddings
        reddit_embs = F.normalize(reddit_embs, p=2, dim=1)
        
        # Calculate Cosine Similarity using highly efficient Matrix Multiplication
        # Shape: (Batch_Size, 768) @ (768, ISOT_Size) -> (Batch_Size, ISOT_Size)
        sim_matrix = torch.matmul(reddit_embs, isot_embs.T)
        
        # Calculate Time Differences: Shape (Batch_Size, ISOT_Size)
        # PyTorch handles this 2D broadcast efficiently without blowing up memory
        time_diffs = torch.abs(batch_times.unsqueeze(1) - isot_timestamps.unsqueeze(0))
        
        # Create a mask: True if within the 3-day window
        time_mask = time_diffs <= TIME_WINDOW_SEC
        
        # Apply the mask: set similarity of out-of-window articles to -1
        sim_matrix = torch.where(time_mask, sim_matrix, torch.tensor(-1.0, device=device))
        
        # Find the best matching ISOT article for each Reddit post
        max_sims, best_indices = torch.max(sim_matrix, dim=1)
        
        # Move results back to CPU for evaluation
        max_sims = max_sims.cpu().numpy()
        best_indices = best_indices.cpu().numpy()
        
        # Check against the strict 0.7 threshold
        for j, sim_score in enumerate(max_sims):
            if sim_score >= SIMILARITY_THRESHOLD:
                matched_post = batch[j]['original_data']
                
                # CRITICAL FIX: Inject the safe name back into the raw payload!
                matched_post['name'] = batch[j]['id'] 
                
                matched_post['ground_truth_label'] = isot_labels[best_indices[j]]
                matched_post['match_similarity'] = float(sim_score)
                matched_results.append(matched_post)
                
    print(f"Successfully matched {len(matched_results)} Reddit posts to ISOT ground truth.")
    return matched_results

def extract_relevant_comments(comments_path, valid_post_ids, output_path):
    print("Extracting associated comments for the network...")
    valid_ids_set = set(valid_post_ids)
    saved_count = 0
    
    with open(comments_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in tqdm(infile, desc="Filtering Comments"):
            c = json.loads(line)
            # Check if the comment belongs to one of our matched posts
            if c.get('link_id') in valid_ids_set:
                outfile.write(line)
                saved_count += 1
                
    print(f"Extracted {saved_count} relevant comments.")

def balance_dataset(posts, label_key="ground_truth_label", max_per_class=None):
    """
    Downsample majority class so labels are balanced.
    """
    class_groups = defaultdict(list)

    for p in posts:
        class_groups[p[label_key]].append(p)

    print("\nPre-balance distribution:")
    for k, v in class_groups.items():
        print(f"Class {k}: {len(v)}")

    # Find minority class size
    if max_per_class is None:
        max_per_class = min(len(v) for v in class_groups.values())

    balanced = []
    for cls, items in class_groups.items():
        random.shuffle(items)
        balanced.extend(items[:max_per_class])

    random.shuffle(balanced)

    print("\nPost-balance distribution:")
    print({k: max_per_class for k in class_groups.keys()})

    return balanced

# --- Execution Flow ---
if __name__ == "__main__":
    # 1. Load and prep data
    isot_titles, isot_times, isot_labels = load_and_prep_isot(ISOT_FILE)
    reddit_posts = filter_reddit_posts(POSTS_FILE)
    
    # 2. Run the matching engine
    matched_posts = match_datasets(reddit_posts, isot_titles, isot_times, isot_labels)

    matched_posts = balance_dataset(matched_posts)
    
    # Extract IDs to filter the comments
    matched_post_ids = [p['name'] for p in matched_posts]
    
    # 3. Save the newly labeled ground truth posts
    print(f"Saving matched posts to {MATCHED_POSTS_OUTPUT}...")
    with open(MATCHED_POSTS_OUTPUT, 'w') as f:
        for post in matched_posts:
            f.write(json.dumps(post) + '\n')
            
    # 4. Extract and save only the comments attached to our matched graph
    extract_relevant_comments(COMMENTS_FILE, matched_post_ids, MATCHED_COMMENTS_OUTPUT)
    
    print("\nStep 1 Complete! You now have a labeled historical dataset ready for the GAT.")