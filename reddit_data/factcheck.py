from collections import defaultdict
import json
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np

# --- Configuration & Paths ---
CLAIMS_FILE = 'claim_review.csv' 
POSTS_FILE = '/data/subreddits25/politics_submissions_2024.jsonl'
MATCHED_POSTS_OUTPUT = 'matched_claims_posts2024.jsonl'
MATCHED_COMMENTS_OUTPUT = 'matched_claims_comments2024.jsonl'
COMMENTS_FILE = '/data/subreddits25/politics_comments_2024.jsonl'

# Filtering Thresholds
SCORE_THRESHOLD = 5         
SIMILARITY_THRESHOLD = 0.50 

def parse_claim_ts(ts_str):
    try:
        clean_ts = ts_str.replace(' UTC', '').strip()
        return datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def load_and_prep_claims(path):
    print("Loading ClaimReview data...")
    # Using sep=None to handle potential tab-separation in your file
    df = pd.read_csv(path, sep=None, engine='python')
    
    # 1. Normalize and clean the rating column
    df['rating_clean'] = df['reviewRating.alternateName'].fillna('').astype(str).str.lower().str.strip()
    
    # 2. STRICT FILTERING
    # We only want rows where the rating is EXACTLY 'true' or 'false'
    # This discards 'mostly true', 'mostly false', 'falso', 'mixture', etc.
    df = df[df['rating_clean'].isin(['true', 'false'])]
    
    # 3. Binary Logic
    # true -> 1, false -> 0
    df['BinaryLabel'] = df['rating_clean'].map({'true': 1, 'false': 0})
    
    # Drop rows missing critical info
    df = df.dropna(subset=['claimReviewed', 'datePublished'])
    
    statements = df['claimReviewed'].tolist()
    labels = df['BinaryLabel'].tolist()
    timestamps = [parse_claim_ts(ts) for ts in df['datePublished'].tolist()]
    
    print(f"Loaded {len(df)} strict claims.")
    print(f"Final Strict Distribution: {labels.count(1)} TRUE (1) | {labels.count(0)} FALSE (0)")
    return statements, labels, timestamps

def filter_reddit_posts(posts_path):
    print(f"Pre-filtering Reddit posts (Min score: {SCORE_THRESHOLD})...")
    filtered_posts = []
    with open(posts_path, 'r') as f:
        for line in tqdm(f, desc="Scanning Posts"):
            try:
                p = json.loads(line)
                post_id = p.get('name') or (f"t3_{p.get('id')}" if p.get('id') else None)
                if post_id and p.get('score', 0) >= SCORE_THRESHOLD:
                    created_dt = datetime.utcfromtimestamp(p.get('created_utc', 0))
                    filtered_posts.append({
                        'id': post_id, 
                        'title': p.get('title', ''),
                        'dt': created_dt,
                        'original_data': p 
                    })
            except: continue
    return filtered_posts

def match_and_label(reddit_posts, c_statements, c_labels, c_timestamps, batch_size=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Encoding Strict Fact-Check claims...")
    c_embs = F.normalize(model.encode(c_statements, convert_to_tensor=True), p=2, dim=1)
    
    candidates = []
    print(f"Matching with 3-Day Window...")
    
    for i in tqdm(range(0, len(reddit_posts), batch_size), desc="Processing Batches"):
        batch = reddit_posts[i:i + batch_size]
        titles = [p['title'] for p in batch]
        reddit_embs = F.normalize(model.encode(titles, convert_to_tensor=True), p=2, dim=1)
        
        sim_matrix = torch.matmul(reddit_embs, c_embs.T)
        
        for row_idx, reddit_post in enumerate(batch):
            similarities = sim_matrix[row_idx].cpu().numpy()
            potential_match_indices = np.where(similarities >= SIMILARITY_THRESHOLD)[0]
            
            best_match = None
            max_sim = -1
            
            for c_idx in potential_match_indices:
                claim_time = c_timestamps[c_idx]
                if not claim_time: continue
                
                # --- 3-DAY TIME CONSTRAINT ---
                time_diff = abs(reddit_post['dt'] - claim_time)
                if time_diff <= timedelta(days=3):
                    if similarities[c_idx] > max_sim:
                        max_sim = similarities[c_idx]
                        best_match = c_idx
            
            if best_match is not None:
                post_data = reddit_post['original_data']
                
                # Assign final label (Strict 1 or 0)
                final_label = int(c_labels[best_match])

                post_data['name'] = reddit_post['id'] 
                post_data['ground_truth_label'] = final_label
                post_data['matched_claim_text'] = c_statements[best_match]
                post_data['claim_published_date'] = str(c_timestamps[best_match])
                post_data['_temp_sim'] = float(max_sim)
                
                candidates.append(post_data)
                
    return candidates

def finalize_top_100(candidates, n=50):
    class_groups = defaultdict(list)
    for c in candidates:
        class_groups[c['ground_truth_label']].append(c)
    
    final_posts = []
    for cls in [0, 1]:
        sorted_cls = sorted(class_groups[cls], key=lambda x: x['_temp_sim'], reverse=True)
        top_n = sorted_cls[:n]
        for item in top_n:
            final_posts.append(item)
    return final_posts

def extract_comments(comments_path, valid_post_ids, output_path):
    ids = set(valid_post_ids)
    print("Extracting Comments...")
    with open(comments_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in tqdm(infile, desc="Scanning Comments"):
            try:
                c = json.loads(line)
                if c.get('link_id') in ids:
                    outfile.write(line)
            except: continue

if __name__ == "__main__":
    import numpy as np
    stmts, lbls, times = load_and_prep_claims(CLAIMS_FILE)
    posts = filter_reddit_posts(POSTS_FILE)
    
    if posts:
        all_matches = match_and_label(posts, stmts, lbls, times)
        
        if all_matches:
            gold_posts = finalize_top_100(all_matches, n=50)
            
            # Save Posts
            with open(MATCHED_POSTS_OUTPUT, 'w') as f:
                for p in gold_posts:
                    f.write(json.dumps(p) + '\n')
            
            # Extract Comments
            extract_comments(COMMENTS_FILE, [p['name'] for p in gold_posts], MATCHED_COMMENTS_OUTPUT)
            
            # Print Final Audit for Verification
            print("\n" + "="*110)
            print(f"{'REDDIT ID':<15} | {'FACT-CHECKED CLAIM':<55} | {'SIM':<5} | {'LABEL'}")
            print("-" * 110)
            for p in gold_posts:
                red_id = p['name']
                claim = (p['matched_claim_text'][:52] + '..') if len(p['matched_claim_text']) > 52 else p['matched_claim_text']
                print(f"{red_id:<15} | {claim:<55} | {p['_temp_sim']:.2f} | {p['ground_truth_label']}")
            print("="*110)
            
            print(f"\nSuccess! Found {len(gold_posts)} high-confidence, strictly binary matches.")
        else:
            print("No matches found within the 3-day window.")