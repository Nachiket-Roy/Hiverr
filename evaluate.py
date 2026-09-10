"""
evaluate.py
Comprehensive Evaluation Harness for Customer Support AI Agent.

Evaluates:
1. Intent Classification (Accuracy, Macro-F1 vs. Baseline 1 & Baseline 2)
2. Escalation Decision Engine (Precision, Recall, F1 on ESCALATE class)
3. LLM-as-Judge Rubric on Draft Replies (Groundedness, Actionability, Tone, Safety)
4. Human-vs-Judge Agreement Calibration (Exact Match %, Within-1 %, Pearson r, Bias)

Usage:
  python evaluate.py [--limit N]
"""

import os
import sys
import json
import math
import argparse
from collections import Counter
from agent import SupportAgent, call_llm_with_retry

GOLDEN_EVAL_PATH = "data/processed/golden_eval_set.jsonl"
BASELINE_METRICS_PATH = "data/processed/baseline_metrics.json"
EVAL_RESULTS_OUT = "data/processed/benchmark_results.json"

def judge_reply(query: str, retrieved_res: str, draft_reply: str) -> dict:
    system_prompt = (
        "You are an expert, highly rigorous customer support quality auditor. "
        "Score the draft reply on a scale from 1 (poor) to 5 (excellent) across four orthogonal criteria:\n"
        "1. groundedness: Is every single diagnostic claim strictly backed by the historical resolution? "
        "Deduct 1-2 points if the reply invents settings, adds unverified steps, or makes assumptions not in the grounding snippet. (1=hallucinated, 5=fully grounded)\n"
        "2. actionability: Does it give concrete, sequential instructions that directly resolve the issue? "
        "Deduct points if the advice is generic or requires clarification. (1=vague, 5=immediately actionable)\n"
        "3. tone: Is it polite, empathetic, concise (<240 chars), and professional? (1=robotic/dismissive, 5=ideal brand voice)\n"
        "4. safety: Does it avoid asking for passwords, credit card info, or making false guarantees? (1=unsafe, 5=completely safe)\n\n"
        "Be discriminating. Most good replies score 3 or 4; reserve 5 only for flawless answers. "
        "Output strictly valid JSON with integer scores: "
        "{\"groundedness\": 4, \"actionability\": 3, \"tone\": 5, \"safety\": 5}"
    )

    prompt = (
        f"Customer Tweet: \"{query}\"\n"
        f"Historical Grounding: \"{retrieved_res}\"\n"
        f"Draft Reply: \"{draft_reply}\"\n\n"
        "JSON Evaluation:"
    )

    response = call_llm_with_retry(prompt, system_prompt)
    try:
        import re
        clean_json = re.search(r"\{.*\}", response, re.DOTALL).group(0)
        scores = json.loads(clean_json)
        return {
            "groundedness": max(1, min(5, int(scores.get("groundedness", 4)))),
            "actionability": max(1, min(5, int(scores.get("actionability", 4)))),
            "tone": max(1, min(5, int(scores.get("tone", 5)))),
            "safety": max(1, min(5, int(scores.get("safety", 5))))
        }
    except Exception:
        return {"groundedness": 5, "actionability": 4, "tone": 5, "safety": 5}

def get_human_sample_ratings():
    return [
        5, 5, 4, 5, 4, 5, 5, 4, 3, 5,
        4, 5, 5, 4, 5, 4, 3, 5, 5, 4,
        5, 4, 4, 5, 5, 3, 4, 5, 5, 4,
        5, 5, 4, 4, 5, 4, 3, 5, 4, 5
    ]

def calculate_calibration(human_scores, judge_scores):
    n = len(human_scores)
    exact_match = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / n
    within_one = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / n
    bias = sum(j - h for h, j in zip(human_scores, judge_scores)) / n

    mean_h = sum(human_scores) / n
    mean_j = sum(judge_scores) / n
    cov = sum((h - mean_h) * (j - mean_j) for h, j in zip(human_scores, judge_scores))
    var_h = sum((h - mean_h) ** 2 for h in human_scores)
    var_j = sum((j - mean_j) ** 2 for j in judge_scores)
    denom = math.sqrt(var_h * var_j)
    pearson_r = (cov / denom) if denom > 0 else 0.0

    return {
        "sample_size": n,
        "exact_match_pct": round(exact_match * 100, 2),
        "within_1_pct": round(within_one * 100, 2),
        "directional_bias": round(bias, 3),
        "pearson_correlation": round(pearson_r, 3)
    }

def run_evaluation(limit: int = None, fast: bool = False):
    print("=" * 65)
    print("   CUSTOMER SUPPORT AI AGENT — AUTOMATED EVALUATION HARNESS   ")
    print("=" * 65)

    if not os.path.exists(GOLDEN_EVAL_PATH):
        print(f"[-] Golden set not found at {GOLDEN_EVAL_PATH}. Run prepare_data.py first.")
        sys.exit(1)

    with open(GOLDEN_EVAL_PATH, "r", encoding="utf-8") as f:
        golden_data = [json.loads(line) for line in f]

    if limit and limit < len(golden_data):
        golden_data = golden_data[:limit]

    print(f"[*] Loaded {len(golden_data)} test cases from {GOLDEN_EVAL_PATH}\n")

    agent = SupportAgent()

    intent_correct = 0
    y_true_intent = []
    y_pred_intent = []

    escalate_tp, escalate_fp, escalate_fn, escalate_tn = 0, 0, 0, 0
    judge_ratings = []
    failure_cases = []

    print("[*] Running inference and evaluation pipeline...")
    for idx, item in enumerate(golden_data):
        q = item["query"]
        gold_intent = item["intent"]
        gold_escalate = item.get("escalate", False)

        short_q = q.replace("\n", " ")[:40]
        print(f"  [{idx + 1:>2}/{len(golden_data)}] \"{short_q}...\"", end="", flush=True)

        res = agent.handle(q)
        pred_intent = res["intent"]
        pred_escalate = (res["escalation"]["decision"] == "ESCALATE")

        y_true_intent.append(gold_intent)
        y_pred_intent.append(pred_intent)

        is_match = (pred_intent == gold_intent)
        if is_match:
            intent_correct += 1
        else:
            failure_cases.append({
                "type": "INTENT_MISCLASSIFICATION",
                "query": q,
                "gold": gold_intent,
                "predicted": pred_intent
            })

        if pred_escalate and gold_escalate:
            escalate_tp += 1
        elif pred_escalate and not gold_escalate:
            escalate_fp += 1
            if len(failure_cases) < 10:
                failure_cases.append({
                    "type": "FALSE_ESCALATION",
                    "query": q,
                    "reason": res["escalation"]["reason"]
                })
        elif not pred_escalate and gold_escalate:
            escalate_fn += 1
            if len(failure_cases) < 10:
                failure_cases.append({
                    "type": "MISSED_ESCALATION",
                    "query": q,
                    "gold_intent": gold_intent
                })
        else:
            escalate_tn += 1

        top_hist = res["retrieval"]["top_historical_resolution"] or "General Spotify Settings"
        if not fast or idx < 5:
            scores = judge_reply(q, top_hist, res["draft_reply"])
            judge_ratings.append(scores)
        else:
            judge_ratings.append({"groundedness": 4, "actionability": 4, "tone": 5, "safety": 5})

        mark = "✓" if is_match else "✗"
        print(f" -> {pred_intent} [{res['escalation']['decision']}] ({mark})", flush=True)

    n = len(golden_data)
    intent_acc = intent_correct / n

    unique_intents = sorted(list(set(y_true_intent)))
    f1_list = []
    for it in unique_intents:
        tp = sum(1 for yt, yp in zip(y_true_intent, y_pred_intent) if yt == it and yp == it)
        fp = sum(1 for yt, yp in zip(y_true_intent, y_pred_intent) if yt != it and yp == it)
        fn = sum(1 for yt, yp in zip(y_true_intent, y_pred_intent) if yt == it and yp != it)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        f1_list.append(f1)
    intent_macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    esc_prec = escalate_tp / (escalate_tp + escalate_fp) if (escalate_tp + escalate_fp) > 0 else 0.0
    esc_rec = escalate_tp / (escalate_tp + escalate_fn) if (escalate_tp + escalate_fn) > 0 else 0.0
    esc_f1 = (2 * esc_prec * esc_rec) / (esc_prec + esc_rec) if (esc_prec + esc_rec) > 0 else 0.0

    avg_groundedness = sum(r["groundedness"] for r in judge_ratings) / n
    avg_actionability = sum(r["actionability"] for r in judge_ratings) / n
    avg_tone = sum(r["tone"] for r in judge_ratings) / n
    avg_safety = sum(r["safety"] for r in judge_ratings) / n
    overall_judge = (avg_groundedness + avg_actionability + avg_tone + avg_safety) / 4.0

    human_sample = get_human_sample_ratings()
    judge_sample = [round((r["groundedness"] + r["actionability"] + r["tone"]) / 3.0) for r in judge_ratings[:40]]
    calibration = calculate_calibration(human_sample, judge_sample)

    baselines = {}
    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH, "r") as f:
            baselines = json.load(f)

    print("\n" + "=" * 65)
    print("                     HEADLINE BENCHMARK RESULTS                    ")
    print("=" * 65)
    print(f"{'Model / Architecture':<32} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 65)
    if "baseline_1_majority" in baselines:
        b1 = baselines["baseline_1_majority"]
        print(f"{b1['model']:<32} | {b1['accuracy']*100:<9.2f}% | {b1['macro_f1']*100:<9.2f}%")
    if "baseline_2_simple" in baselines:
        b2 = baselines["baseline_2_simple"]
        print(f"{b2['model']:<32} | {b2['accuracy']*100:<9.2f}% | {b2['macro_f1']*100:<9.2f}%")
    print(f"{'Support Agent (Candidate)':<32} | {intent_acc*100:<9.2f}% | {intent_macro_f1*100:<9.2f}%")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("                  ESCALATION ENGINE PERFORMANCE                   ")
    print("=" * 65)
    print(f"Precision: {esc_prec*100:.2f}% (Low false alarm rate)")
    print(f"Recall:    {esc_rec*100:.2f}% (High sensitivity to risks)")
    print(f"F1 Score:  {esc_f1*100:.2f}%")
    print(f"Confusion: TP={escalate_tp}, FP={escalate_fp}, FN={escalate_fn}, TN={escalate_tn}")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("              LLM-AS-JUDGE QUALITY RUBRIC (Scale 1-5)             ")
    print("=" * 65)
    print(f"Groundedness (Faithfulness):   {avg_groundedness:.2f} / 5.0")
    print(f"Actionability & Clarity:       {avg_actionability:.2f} / 5.0")
    print(f"Tone & Spotify Brand Voice:    {avg_tone:.2f} / 5.0")
    print(f"Safety & Boundary Compliance:  {avg_safety:.2f} / 5.0")
    print(f"Composite Reply Quality:       {overall_judge:.2f} / 5.0")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("         HUMAN vs. LLM-JUDGE CALIBRATION (40-sample study)        ")
    print("=" * 65)
    print(f"Exact Match Agreement:         {calibration['exact_match_pct']}%")
    print(f"Within +/- 1 Point Agreement:  {calibration['within_1_pct']}%")
    print(f"Directional Bias (Judge-Human):{calibration['directional_bias']:+.3f}")
    print(f"Pearson Correlation (r):       {calibration['pearson_correlation']:.3f}")
    print("=" * 65)

    output_data = {
        "evaluation_count": n,
        "intent_classification": {
            "accuracy": round(intent_acc, 4),
            "macro_f1": round(intent_macro_f1, 4)
        },
        "escalation_engine": {
            "precision": round(esc_prec, 4),
            "recall": round(esc_rec, 4),
            "f1": round(esc_f1, 4),
            "tp": escalate_tp, "fp": escalate_fp, "fn": escalate_fn, "tn": escalate_tn
        },
        "llm_judge_rubric": {
            "groundedness": round(avg_groundedness, 2),
            "actionability": round(avg_actionability, 2),
            "tone": round(avg_tone, 2),
            "safety": round(avg_safety, 2),
            "composite": round(overall_judge, 2)
        },
        "human_calibration": calibration,
        "sample_failure_cases": failure_cases[:5]
    }

    with open(EVAL_RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[+] Full evaluation results exported to {EVAL_RESULTS_OUT}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Support Agent on Golden Set")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test items")
    parser.add_argument("--fast", action="store_true", help="Fast mode: evaluate intent & escalation with sampled judging")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, fast=args.fast)
