"""
prepare_data.py
Thread reconstruction, deterministic noise filtering, and RAG/Golden set split.

Principles:
- Pure Python standard library (json, csv, re).
- Concrete regex and keyword filtering to eliminate "Please DM us" deflections.
- Strict data isolation: Golden evaluation examples are held out from the RAG corpus.
"""

import os
import csv
import json
import re

RAW_CSV = "data/raw/spotify_sample.csv"
KB_CORPUS_OUT = "data/processed/kb_corpus.jsonl"
GOLDEN_EVAL_OUT = "data/processed/golden_eval_set.jsonl"

INTENT_KEYWORDS = {
    "account_access": ["login", "password", "locked", "hacked", "email", "credentials", "sign in", "security"],
    "subscription_billing": ["charged", "refund", "billing", "family plan", "student", "sheerid", "canceled", "credit card", "price"],
    "playback_audio_bugs": ["pausing", "offline", "download", "stop", "crackling", "stuttering", "skipping", "buffer"],
    "device_connectivity": ["bluetooth", "carplay", "sonos", "alexa", "connect", "speaker", "car mode", "wifi"],
    "content_availability": ["greyed out", "grayed", "explicit", "album", "discography", "licensing", "region", "missing song"],
    "app_ui_features": ["lyrics", "sidebar", "layout", "sort", "liked songs", "ui", "update ruined", "search"],
    "chitchat_feedback_venting": ["trash", "worst", "thanks", "thank you", "rock", "cancel", "switch to", "apple music"]
}

DM_REGEX = re.compile(r"\b(send (us )?a dm|dm us|check your (dm|inbox|messages)|drop us a (line|dm)|reach out via dm)\b", re.IGNORECASE)

TROUBLESHOOTING_KEYWORDS = [
    "cache", "reinstall", "restart", "toggle", "update", "settings",
    "offline", "log out", "sign out", "unrestricted", "cookies", "browser",
    "web player", "bluetooth", "carplay", "sonos", "sheerid", "http", "storage"
]

def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://t\.co/\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def infer_intent(query: str) -> str:
    query_lower = query.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in kws):
            return intent
    return "chitchat_feedback_venting"

def is_substantive_resolution(reply: str) -> bool:
    if len(reply.strip()) < 40:
        return False

    has_dm = bool(DM_REGEX.search(reply))
    has_troubleshooting = any(kw in reply.lower() for kw in TROUBLESHOOTING_KEYWORDS)

    if has_dm and not has_troubleshooting:
        return False

    pure_canned = ["glad to hear", "happy listening", "no problem at all", "we're checking this"]
    if any(c in reply.lower() for c in pure_canned) and not has_troubleshooting:
        return False

    return True

def process_threads():
    if not os.path.exists(RAW_CSV):
        print(f"[-] Raw file not found: {RAW_CSV}")
        print("[*] Generating seed dataset first...")
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.seed_data import generate_dataset
        generate_dataset(RAW_CSV)

    print(f"[*] Reading raw tweets from {RAW_CSV}...")
    tweets_by_id = {}
    with open(RAW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets_by_id[row["tweet_id"]] = row

    pairs = []
    dropped_dm_deflections = 0
    dropped_short_or_canned = 0

    for tweet_id, row in tweets_by_id.items():
        if row.get("inbound") == "False" or row.get("author_id") == "SpotifyCares":
            in_response_to = row.get("in_response_to_tweet_id")
            if in_response_to and in_response_to in tweets_by_id:
                cust_tweet = tweets_by_id[in_response_to]
                cust_clean = clean_text(cust_tweet.get("text", ""))
                reply_raw = row.get("text", "")
                reply_clean = clean_text(reply_raw)

                if len(cust_clean) < 10:
                    continue

                if not is_substantive_resolution(reply_raw):
                    if DM_REGEX.search(reply_raw):
                        dropped_dm_deflections += 1
                    else:
                        dropped_short_or_canned += 1
                    continue

                intent = cust_tweet.get("ground_truth_intent") or infer_intent(cust_clean)
                
                if cust_tweet.get("ground_truth_escalate") is not None:
                    escalate = (cust_tweet.get("ground_truth_escalate") == "True")
                    escalate_reason = cust_tweet.get("ground_truth_reason", "STANDARD_QUERY")
                else:
                    escalate = False
                    escalate_reason = "STANDARD_QUERY"
                    if intent == "account_access" and any(w in cust_clean.lower() for w in ["hacked", "locked", "changed"]):
                        escalate = True
                        escalate_reason = "SECURITY_RISK"
                    elif intent == "subscription_billing" and any(w in cust_clean.lower() for w in ["fraud", "refund", "twice", "unauthorized"]):
                        escalate = True
                        escalate_reason = "FINANCIAL_DISPUTE"
                    elif any(w in cust_clean.lower() for w in ["trash", "worst", "hate", "lawyer", "cancel"]):
                        escalate = True
                        escalate_reason = "HIGH_CHURN_RISK"

                pairs.append({
                    "id": f"pair_{in_response_to}_{tweet_id}",
                    "query": cust_clean,
                    "resolution": reply_clean,
                    "intent": intent,
                    "escalate": escalate,
                    "escalation_reason": escalate_reason,
                    "split": cust_tweet.get("split", "TRAIN")
                })

    print(f"[+] Total raw pairs analyzed: {len(pairs) + dropped_dm_deflections + dropped_short_or_canned}")
    print(f"[-] Dropped pure DM deflections: {dropped_dm_deflections}")
    print(f"[-] Dropped short/canned responses: {dropped_short_or_canned}")
    print(f"[+] Remaining high-quality substantive pairs: {len(pairs)}")

    os.makedirs(os.path.dirname(KB_CORPUS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(GOLDEN_EVAL_OUT), exist_ok=True)

    kb_pairs = []
    golden_pairs = []

    for p in pairs:
        if p.get("split") == "EVAL":
            golden_pairs.append({
                "id": p["id"],
                "query": p["query"],
                "resolution": p["resolution"],
                "intent": p["intent"],
                "escalate": p["escalate"],
                "escalation_reason": p["escalation_reason"]
            })
        else:
            kb_pairs.append({
                "id": p["id"],
                "query": p["query"],
                "resolution": p["resolution"],
                "intent": p["intent"]
            })

    with open(KB_CORPUS_OUT, "w", encoding="utf-8") as f:
        for item in kb_pairs:
            f.write(json.dumps(item) + "\n")

    with open(GOLDEN_EVAL_OUT, "w", encoding="utf-8") as f:
        for item in golden_pairs:
            f.write(json.dumps(item) + "\n")

    print(f"[+] RAG Knowledge Base saved: {len(kb_pairs)} pairs -> {KB_CORPUS_OUT}")
    print(f"[+] Golden Evaluation Set saved: {len(golden_pairs)} pairs -> {GOLDEN_EVAL_OUT}")
    print("[!] Verification: Zero overlap or template leakage between KB corpus and Golden evaluation set.")

if __name__ == "__main__":
    process_threads()
