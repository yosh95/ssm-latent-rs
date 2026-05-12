#!/usr/bin/env python3
"""NAB evaluation script using the official NAB Runner pipeline.

This script takes the pre-generated detector results (from the Rust SSM+Latent
implementation) and runs the official NAB optimize → score → normalize pipeline
to produce final benchmark scores.
"""
import os
import sys
try:
  import simplejson as json
except ImportError:
  import json

# Add current directory to path so we can import nab
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nab.runner import Runner
from nab.corpus import Corpus
from nab.labeler import CorpusLabel
from nab.optimizer import optimizeThreshold
from nab.scorer import scoreCorpus
from nab.util import convertResultsPathToDataPath

def main():
    detector = "ssm_latent"
    root = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(root, "data")
    results_dir = os.path.join(root, "results")
    windows_file = os.path.join(root, "labels", "combined_windows.json")
    profiles_file = os.path.join(root, "config", "profiles.json")
    thresholds_file = os.path.join(root, "config", "thresholds.json")
    
    print(f"\n=== NAB Evaluation for detector: {detector} ===")
    print(f"Data dir:    {data_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Windows:     {windows_file}")
    
    # Use the official Runner which handles optimize → score → normalize
    runner = Runner(
        dataDir=data_dir,
        labelPath=windows_file,
        resultsDir=results_dir,
        profilesPath=profiles_file,
        thresholdPath=thresholds_file,
        numCPUs=1,
    )
    runner.initialize()
    
    # Step 1: Optimize thresholds
    print("\n[Step 1] Optimizing thresholds...")
    thresholds = runner.optimize([detector])
    
    # Save thresholds
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to {thresholds_file}")
    
    # Step 2: Score with optimized thresholds
    print("\n[Step 2] Scoring...")
    runner.score([detector], thresholds)
    
    # Step 3: Normalize
    print("\n[Step 3] Normalizing...")
    try:
        runner.normalize()
    except Exception as e:
        print(f"Normalization skipped (null detector baseline not available): {e}")
        # Print raw scores instead
        for profile_name in runner.profiles:
            score_file = os.path.join(results_dir, detector, f"{detector}_{profile_name}_scores.csv")
            if os.path.exists(score_file):
                import pandas
                df = pandas.read_csv(score_file)
                total_score = df["Score"].iloc[-1]
                print(f"  Raw score ({profile_name}): {total_score:.4f}")
    
    print("\n=== Evaluation complete ===")

if __name__ == "__main__":
    main()
