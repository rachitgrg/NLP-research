import os
import csv
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nlp.keyword_extractor import load_nlp_model
from nlp.query_parser import parse_query

def evaluate_semantics(gold_path: str, results_path: str):
    if not os.path.exists(gold_path) or not os.path.exists(results_path):
        print("Required CSV files missing. Run evaluate_models.py first.")
        return

    # Load gold
    golds = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            golds[row["audio_file"]] = {
                "intent": row["intent"],
                "object": row["object"],
                "color": row["color"] or None,
                "size": row["size"] or None,
                "position": row["position"] or None
            }

    # Load results
    asr_results = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asr_results.append(row)

    print("Loading NLP model...")
    nlp = load_nlp_model()

    models_to_eval = ["reference", "sarvam", "whisper_tiny", "whisper_base"]
    stats = {m: {"intent": 0, "object": 0, "attribute": 0, "complete": 0, "total": 0} for m in models_to_eval}

    print("\nEvaluating Semantic Accuracy...")

    for row in asr_results:
        sample_id = row["sample_id"]
        gold = golds.get(sample_id)
        if not gold:
            continue

        for m in models_to_eval:
            text = row["reference"] if m == "reference" else row.get(f"{m}_text", "")
            if not text:
                parsed = {"intent": "unknown", "object": None, "attributes": {}}
            else:
                parsed = parse_query(nlp, text)
            
            # Compare
            intent_ok = parsed["intent"] == gold["intent"]
            object_ok = parsed["object"] == gold["object"]
            
            # Attributes
            parsed_attrs = parsed.get("attributes", {})
            color_ok = parsed_attrs.get("color") == gold["color"]
            size_ok = parsed_attrs.get("size") == gold["size"]
            pos_ok = parsed_attrs.get("position") == gold["position"]
            attr_ok = color_ok and size_ok and pos_ok
            
            complete_ok = intent_ok and object_ok and attr_ok
            
            stats[m]["total"] += 1
            if intent_ok: stats[m]["intent"] += 1
            if object_ok: stats[m]["object"] += 1
            if attr_ok: stats[m]["attribute"] += 1
            if complete_ok: stats[m]["complete"] += 1

    print("=" * 50)
    print("SEMANTIC EVALUATION RESULTS")
    print("=" * 50)
    
    for m in models_to_eval:
        s = stats[m]
        tot = max(1, s["total"])
        print(f"--- {m.upper()} ---")
        print(f"Intent Accuracy:    {(s['intent'] / tot) * 100:.1f}%")
        print(f"Object Accuracy:    {(s['object'] / tot) * 100:.1f}%")
        print(f"Attribute Accuracy: {(s['attribute'] / tot) * 100:.1f}%")
        print(f"Complete Accuracy:  {(s['complete'] / tot) * 100:.1f}%\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(base_dir, "semantic_gold.csv")
    results_path = os.path.join(base_dir, "results.csv")
    evaluate_semantics(gold_path, results_path)
