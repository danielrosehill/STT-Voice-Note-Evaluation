#!/usr/bin/env python3
"""
STT Evaluation Script

This script compares STT API results against ground truth transcripts
and calculates accuracy metrics like Word Error Rate (WER).

Usage:
    python evaluate_stt.py --ground-truth texts/ --results results/whisper/
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import difflib

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Use difflib to find edit distance
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    
    # Count operations needed
    operations = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            operations += max(i2 - i1, j2 - j1)
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')
    
    return operations / len(ref_words)

def load_ground_truth(ground_truth_dir: str) -> Dict[str, str]:
    """Load ground truth transcripts from directory."""
    ground_truth = {}
    
    for file_path in Path(ground_truth_dir).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            ground_truth[file_path.stem] = content
    
    return ground_truth

def load_stt_results(results_dir: str) -> Dict[str, str]:
    """Load STT results from directory."""
    results = {}
    
    for file_path in Path(results_dir).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            results[file_path.stem] = content
    
    return results

def evaluate_stt_service(ground_truth: Dict[str, str], 
                        stt_results: Dict[str, str], 
                        service_name: str) -> Dict:
    """Evaluate a single STT service against ground truth."""
    
    results = {
        'service': service_name,
        'total_samples': 0,
        'total_wer': 0.0,
        'individual_scores': {}
    }
    
    for file_id, reference in ground_truth.items():
        if file_id in stt_results:
            hypothesis = stt_results[file_id]
            wer = calculate_wer(reference, hypothesis)
            
            results['individual_scores'][file_id] = {
                'wer': wer,
                'reference_words': len(reference.split()),
                'hypothesis_words': len(hypothesis.split())
            }
            
            results['total_wer'] += wer
            results['total_samples'] += 1
    
    if results['total_samples'] > 0:
        results['average_wer'] = results['total_wer'] / results['total_samples']
    else:
        results['average_wer'] = 0.0
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate STT accuracy')
    parser.add_argument('--ground-truth', required=True, 
                       help='Directory containing ground truth transcripts')
    parser.add_argument('--results', required=True,
                       help='Directory containing STT results')
    parser.add_argument('--service-name', default='Unknown',
                       help='Name of the STT service being evaluated')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    
    print(f"Loading STT results from {args.results}")
    stt_results = load_stt_results(args.results)
    
    # Evaluate
    print(f"Evaluating {args.service_name}")
    evaluation = evaluate_stt_service(ground_truth, stt_results, args.service_name)
    
    # Print results
    print(f"\nResults for {evaluation['service']}:")
    print(f"Samples evaluated: {evaluation['total_samples']}")
    print(f"Average WER: {evaluation['average_wer']:.3f}")
    print(f"Average accuracy: {(1 - evaluation['average_wer']) * 100:.1f}%")
    
    print("\nIndividual file scores:")
    for file_id, score in evaluation['individual_scores'].items():
        accuracy = (1 - score['wer']) * 100
        print(f"  {file_id}: {accuracy:.1f}% accuracy (WER: {score['wer']:.3f})")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
