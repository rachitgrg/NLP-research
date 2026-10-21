import os
import pytest
from evaluation.semantic_eval import evaluate_semantics

def test_evaluate_semantics(capsys, tmp_path):
    # This test simply checks that the script runs without crashing when given valid paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_path = os.path.join(base_dir, "evaluation", "semantic_gold.csv")
    results_path = os.path.join(base_dir, "evaluation", "results.csv")
    
    if os.path.exists(gold_path) and os.path.exists(results_path):
        evaluate_semantics(gold_path, results_path)
        captured = capsys.readouterr()
        assert "SEMANTIC EVALUATION RESULTS" in captured.out
        assert "REFERENCE" in captured.out
