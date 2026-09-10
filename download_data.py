"""
download_data.py
Data lineage and extraction script for Twitter Customer Support Dataset.

Usage:
  1. Download twcs.csv from Kaggle:
     kaggle datasets download -d thoughtvector/customer-support-on-twitter -f twcs.csv
     unzip twcs.csv.zip -d data/raw/

  2. Run streaming extractor to slice Spotify interactions:
     python download_data.py --input data/raw/twcs.csv --output data/raw/spotify_sample.csv

If twcs.csv is not present, seed_data.py will automatically generate a clean
subsample for immediate execution.
"""

import os
import csv
import sys
import argparse

TARGET_BRAND = "SpotifyCares"

def extract_brand_tweets(input_path: str, output_path: str, max_rows: int = 5000):
    """
    Streams twcs.csv to extract rows involving TARGET_BRAND without
    loading the full 3-million-row file into memory.
    """
    if not os.path.exists(input_path):
        print(f"[-] Input file not found: {input_path}")
        print("[!] To download the raw dataset:")
        print("    kaggle datasets download -d thoughtvector/customer-support-on-twitter -f twcs.csv")
        return False

    print(f"[*] Streaming {input_path} to find {TARGET_BRAND} interactions...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    extracted = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        
        with open(output_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                author = row.get("author_id", "")
                text = row.get("text", "")

                if author == TARGET_BRAND or f"@{TARGET_BRAND}" in text:
                    writer.writerow(row)
                    extracted += 1
                    if extracted >= max_rows:
                        break

    print(f"[+] Successfully extracted {extracted} rows to {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Spotify support tweets from twcs.csv")
    parser.add_argument("--input", default="data/raw/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output", default="data/raw/spotify_sample.csv", help="Path to save output sample")
    parser.add_argument("--max_rows", type=int, default=5000, help="Maximum rows to extract")
    args = parser.parse_args()

    success = extract_brand_tweets(args.input, args.output, args.max_rows)
    if not success:
        sys.exit(1)
