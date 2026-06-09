"""
Reddit Submissions Parser
Parses a JSONL file of Reddit submissions, fetches AskNews reliability scores,
and outputs a new JSONL file with 'source_score' and 'missing_source_rate' added.
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def load_asknews_interface(filepath: str):
    """Dynamically load the interface file since it has hyphens in the name."""
    path = Path(filepath)
    if not path.exists():
        log.error(f"Could not find the interface file at: {filepath}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("asknews_interface", str(path))
    asknews_interface = importlib.util.module_from_spec(spec)
    sys.modules["asknews_interface"] = asknews_interface
    spec.loader.exec_module(asknews_interface)
    return asknews_interface


def main():
    parser = argparse.ArgumentParser(description="Parse Reddit JSONL and append AskNews scores.")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-api-calls", type=int, default=50, help="Maximum number of AskNews API calls to make")
    parser.add_argument("--interface-path", type=str, default="app/asknews/asknews-adfontes-interface-2.py", help="Path to the AskNews interface script")
    args = parser.parse_args()

    # Load the custom module
    asknews = load_asknews_interface(args.interface_path)
    
    # Override the hardcoded limit in the interface file so they stay perfectly in sync
    asknews.MAX_API_CALLS = args.max_api_calls

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    api_calls_made = 0
    cache_hits = 0
    lines_processed = 0
    errors = 0

    log.info(f"Starting parsing. Max API calls allowed: {args.max_api_calls}")

    # Process line-by-line for memory efficiency
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                log.warning(f"Skipping invalid JSON line: {line[:50]}...")
                continue

            # Extract required fields
            post_id = data.get("id")
            title = data.get("title")
            created_utc = data.get("created_utc")

            # Default the new fields to null (None) in case of failure or limits
            data["source_score"] = None
            data["missing_source_rate"] = None

            if not all([post_id, title, created_utc]):
                log.debug(f"Skipping post {post_id}: missing required fields.")
                outfile.write(json.dumps(data) + "\n")
                continue

            # Determine if we have API credits left
            allow_api = api_calls_made < args.max_api_calls

            try:
                # Call our scoring interface
                result = asknews.score_post(
                    post_id=post_id,
                    post_title=title,
                    post_time=created_utc,
                    allow_api_call=allow_api
                )

                # Assign the calculated scores
                data["source_score"] = result["source_score"]
                data["missing_source_rate"] = result["missing_source_rate"]

                # Track metrics
                if result["is_cached"]:
                    cache_hits += 1
                else:
                    api_calls_made += 1

            except asknews.ApiLimitReachedError:
                # We expected this if limit is reached and it's not in cache. Just move on.
                pass
            except asknews.NoArticlesFoundError:
                # API was called, but no articles had extractable domains. Counts as an API call.
                if allow_api:  # We assume if allow_api was True, it was a cache miss that threw this
                    api_calls_made += 1
                errors += 1
            except Exception as e:
                log.warning(f"Unexpected error on post {post_id}: {e}")
                errors += 1

            # Write the updated JSON object as a new line
            outfile.write(json.dumps(data) + "\n")
            lines_processed += 1

            # Print progress every 100 lines
            if lines_processed % 100 == 0:
                log.info(f"Processed {lines_processed} lines... (API Calls: {api_calls_made}/{args.max_api_calls}, Cache Hits: {cache_hits})")

    # Final summary
    log.info("=== Parsing Complete ===")
    log.info(f"Total lines processed: {lines_processed}")
    log.info(f"Total cache hits:      {cache_hits}")
    log.info(f"Total API calls made:  {api_calls_made}")
    log.info(f"Total posts un-scored: {errors} (due to no articles/domains)")
    log.info(f"Output saved to:       {output_path}")

if __name__ == "__main__":
    main()