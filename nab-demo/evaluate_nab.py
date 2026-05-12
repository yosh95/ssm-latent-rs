#!/usr/bin/env python3
import os
import sys
import argparse
import json

# Add current directory to path so we can import nab
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nab.scorer import ScoringWeights, TabularScorer
from nab.corpus import Corpus
from nab.labeler import CorpusLabel

def main():
    parser = argparse.ArgumentParser(description="NAB Scorer Script")
    parser.add_argument("--resultsDir", default="results", help="Directory containing result CSVs")
    parser.add_argument("--labelsDir", default="labels", help="Directory containing combined_windows.json")
    parser.add_argument("--dataDir", default="data", help="Directory containing raw data CSVs")
    parser.add_argument("--detector", default="ssm_latent", help="Name of the detector to score")
    
    args = parser.parse_args()

    # Define paths
    windows_path = os.path.join(args.labelsDir, "combined_windows.json")
    
    print(f"Loading corpus from {args.dataDir}...")
    corpus = Corpus(args.dataDir)
    
    print(f"Loading labels from {windows_path}...")
    corpus_label = CorpusLabel(windows_path, corpus)
    
    # Define profiles
    profiles = {
        "Standard": ScoringWeights(tpWeight=1.0, fpWeight=0.11, fnWeight=1.0, tnWeight=0.0),
        "Reward Low FP": ScoringWeights(tpWeight=1.0, fpWeight=0.22, fnWeight=1.0, tnWeight=0.0),
        "Reward Low FN": ScoringWeights(tpWeight=1.0, fpWeight=0.11, fnWeight=2.0, tnWeight=0.0),
    }

    scorer = TabularScorer(corpus, corpus_label, profiles, verbosity=1)
    
    print(f"Scoring results for detector: {args.detector}...")
    # Results are expected at results/<detector>/<category>/<file>.csv
    # We will score all files found in the detector's directory
    detector_dir = os.path.join(args.resultsDir, args.detector)
    
    if not os.path.exists(detector_dir):
        print(f"Error: Results directory not found: {detector_dir}")
        return

    # Add detections from result files
    for root, _, files in os.walk(detector_dir):
        for f in files:
            if f.endswith(".csv"):
                rel_path = os.path.relpath(os.path.join(root, f), detector_dir)
                result_path = os.path.join(root, f)
                print(f"Adding results from {rel_path}...")
                scorer.addDetectorResults(args.detector, rel_path, result_path)

    # Compute scores
    print("\nComputing scores...")
    scorer.score()
    
    # Get scores
    results = scorer.getScores()
    
    print("\n" + "="*40)
    print(f" NAB SCORES: {args.detector}")
    print("="*40)
    for profile_name, profile_results in results[args.detector].items():
        score = profile_results["score"]
        print(f" {profile_name:15}: {score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
