"""
Misinformation Research Pipeline
Queries AskNews for related articles and scores source reliability using Ad Fontes Media.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from asknews_sdk import AskNewsSDK
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION & FILE PATHS
# ==========================================
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
OLD_CACHE_FILE = CACHE_DIR / "asknews_cache.json"  # Retained for backwards compatibility
CACHE_FILE = CACHE_DIR / "asknews_cache.jsonl"     # New append-only JSON Lines cache
ADFONTES_PATH = BASE_DIR.parent.parent / "resources" / "adfontes-2026-02-06.json"

NEUTRAL_RELIABILITY = 32.0  # midpoint of 0-64 scale
MAX_API_CALLS = 50          # Safeguard to protect your credits

# Global state to prevent redundant disk I/O and auth loops
_client_instance: AskNewsSDK | None = None
_api_calls_made = 0
_cache_memory: dict | None = None
_adfontes_lookup: dict[str, dict] | None = None


# ==========================================
# EXCEPTIONS
# ==========================================
class ApiLimitReachedError(Exception):
    pass

class NoArticlesFoundError(Exception):
    pass


# ==========================================
# 1. ASKNEWS API & CACHING
# ==========================================
def _get_client() -> AskNewsSDK:
    global _client_instance
    if _client_instance is None:
        _client_instance = AskNewsSDK(
            client_id=os.environ["ASKNEWS_CLIENT_ID"],
            client_secret=os.environ["ASKNEWS_CLIENT_SECRET"],
        )
    return _client_instance

def _load_cache() -> dict:
    global _cache_memory
    if _cache_memory is not None:
        return _cache_memory

    _cache_memory = {}

    # 1. Load legacy JSON cache if it exists (so you don't lose previous work)
    if OLD_CACHE_FILE.exists():
        try:
            with open(OLD_CACHE_FILE, "r") as f:
                legacy_data = json.load(f)
                if isinstance(legacy_data, dict):
                    _cache_memory.update(legacy_data)
        except Exception as e:
            log.warning(f"Failed to load legacy cache: {e}")

    # 2. Load the new JSONL append-only cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        _cache_memory.update(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.warning(f"Failed to load JSONL cache: {e}")

    return _cache_memory

def _append_to_cache(post_id: str, data: dict) -> None:
    global _cache_memory
    if _cache_memory is None:
        _cache_memory = _load_cache()
        
    _cache_memory[post_id] = data
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Append exactly one new line to the file
    with open(CACHE_FILE, "a") as f:
        json.dump({post_id: data}, f, default=str)
        f.write("\n")

def query_asknews(post_id: str, post_title: str, post_time: float, search_range: float = 48.0, allow_api_call: bool = True) -> tuple[list[dict], bool]:
    global _api_calls_made
    cache = _load_cache()

    if post_id in cache:
        return cache[post_id]["articles"], True

    if not allow_api_call:
        raise ApiLimitReachedError("API call required but allow_api_call is False.")
    if _api_calls_made >= MAX_API_CALLS:
        raise ApiLimitReachedError(f"Hard limit of {MAX_API_CALLS} API calls reached.")

    client = _get_client()
    
    # Calculate the time window in seconds
    half_window_seconds = (search_range / 2) * 3600
    start_timestamp = int(post_time - half_window_seconds)
    end_timestamp = int(post_time + half_window_seconds)

    response = client.news.search_news(
        query=post_title,
        n_articles=30,
        return_type="dicts",
        method="nl",
        historical=True,
        similarity_score_threshold=0.8,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
    )

    articles = [a.model_dump(mode="json") for a in response.as_dicts]
    _api_calls_made += 1

    # Append to file instead of rewriting it
    cache_data = {
        "query": post_title,
        "post_time_epoch": post_time,
        "search_range_hours": search_range,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "articles": articles,
    }
    _append_to_cache(post_id, cache_data)

    return articles, False


# ==========================================
# 2. DOMAIN EXTRACTION (FALLBACK STRATEGY)
# ==========================================
COUNTRY_CODE_TLDS = {
    "co.uk", "co.nz", "com.au", "com.mx", "com.ar",
    "co.za", "co.in", "co.jp", "co.kr", "com.br",
    "com.cn", "com.tr", "com.ua", "com.sg", "com.hk",
    "co.il", "org.uk", "org.au",
}
KEEP_SUBDOMAIN_DOMAINS = {"substack.com"}

def get_domain_variations(domain_url: str) -> list[str]:
    if not domain_url:
        return []

    hostname = domain_url.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Attempt 1 & 2: Exact hostname (e.g., "sg.news.yahoo.com" and ".sg.news.yahoo.com")
    variations = [hostname, f".{hostname}"]

    parts = hostname.split(".")
    if len(parts) < 2:
        return variations

    root_2 = ".".join(parts[-2:])
    
    # If it's a domain where we keep the subdomain (like substack), stop here
    if root_2 in KEEP_SUBDOMAIN_DOMAINS:
        return variations

    # Otherwise, calculate the root domain
    if len(parts) >= 3 and root_2 in COUNTRY_CODE_TLDS:
        root = ".".join(parts[-3:])
    else:
        root = root_2

    # Attempt 3 & 4: Root domain fallback (e.g., "yahoo.com" and ".yahoo.com")
    if root != hostname:
        variations.extend([root, f".{root}"])

    return variations


# ==========================================
# 3. AD FONTES RELIABILITY LOOKUP
# ==========================================
def load_adfontes(path: Path = ADFONTES_PATH) -> dict[str, dict]:
    global _adfontes_lookup
    if _adfontes_lookup is not None:
        return _adfontes_lookup

    with open(path) as f:
        data = json.load(f)

    sources = data["props"]["pageProps"]["jsonData"]["sources"]
    _adfontes_lookup = {
        s["domain"]: {
            "reliability_mean": s["reliability_mean"],
            "bias_mean": s["bias_mean"],
            "moniker_name": s["moniker_name"],
        }
        for s in sources
    }
    return _adfontes_lookup

def lookup_reliability(domain_url: str) -> tuple[float, bool, str]:
    lookup = load_adfontes()
    variations = get_domain_variations(domain_url)
    
    for variant in variations:
        entry = lookup.get(variant)
        if entry is not None:
            return entry["reliability_mean"], True, variant

    return NEUTRAL_RELIABILITY, False, domain_url


# ==========================================
# 4. SCORING & OUTPUT PIPELINE
# ==========================================
def compute_source_score(articles: list[dict]) -> dict:
    if not articles:
        raise NoArticlesFoundError("AskNews returned zero articles for this post")

    total, num_unrated = 0, 0
    reliability_sum = 0.0

    for article in articles:
        raw_domain = article.get("domain_url") 
        if not raw_domain:
            continue

        reliability, is_rated, _ = lookup_reliability(raw_domain)
        reliability_sum += reliability / 64.0  # normalize to 0-1
        total += 1
        if not is_rated:
            num_unrated += 1

    if total == 0:
        raise NoArticlesFoundError("No articles with extractable domains.")

    return {
        "source_score": reliability_sum / total,
        "missing_source_rate": num_unrated / total,
        "num_articles": total,
        "num_unrated": num_unrated,
    }

def get_related_articles(articles: list[dict]) -> list[dict]:
    rated = []
    for article in articles:
        raw_domain = article.get("domain_url") 
        if not raw_domain:
            continue
            
        reliability, is_rated, matched_domain = lookup_reliability(raw_domain)
        if not is_rated:
            continue

        rated.append({
            "title": article.get("eng_title") or article.get("title", ""),
            "url": article.get("article_url", ""),
            "source_name": article.get("source_id", ""),
            "reliability_score": reliability,
            "domain": matched_domain,
        })

    rated.sort(key=lambda x: x["reliability_score"], reverse=True)
    return rated

def score_post(post_id: str, post_title: str, post_time: float, search_range: float = 48.0, allow_api_call: bool = True) -> dict:
    """
    Score a post's reliability based on related AskNews articles.
    
    Args:
        post_id: Unique identifier (e.g., Reddit post ID).
        post_title: The query to search.
        post_time: UTC epoch timestamp of the post.
        search_range: Total search window in hours (default 48.0 splits to +/- 24 hours).
        allow_api_call: If False, only searches the local cache.
    """
    articles, is_cached = query_asknews(post_id, post_title, post_time, search_range, allow_api_call)
    
    result = compute_source_score(articles)
    result["related_articles"] = get_related_articles(articles)
    result["is_cached"] = is_cached
    return result

if __name__ == "__main__":
    try:
        # Example run using the current time as the "post_time"
        current_time_epoch = datetime.now(timezone.utc).timestamp()
        
        res = score_post(
            post_id="1paxoib", 
            post_title="House and Senate committees launch inquiries into second strike on alleged drug boat",
            post_time=1764547334.0,
            search_range=48.0
        )
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")