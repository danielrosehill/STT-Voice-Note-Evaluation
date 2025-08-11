#!/usr/bin/env python3
"""
Speaking Rate Analysis for STT Evaluation

This script calculates speaking rates (words per minute) for each voice note sample
and correlates them with STT model accuracy to identify performance patterns.

Usage:
    python speaking_rate_analysis.py --dataset ../data-manifest/english_dataset.json --results ../results/openai_comprehensive_evaluation_raw.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import wave
import contextlib

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0

def calculate_speaking_rate(text: str, duration_seconds: float) -> float:
    """Calculate speaking rate in words per minute."""
    if duration_seconds == 0:
        return 0.0
    
    word_count = len(text.split())
    duration_minutes = duration_seconds / 60.0
    return word_count / duration_minutes if duration_minutes > 0 else 0.0

def categorize_speaking_rate(wpm: float) -> str:
    """Categorize speaking rate into descriptive categories."""
    if wpm < 120:
        return "slow"
    elif wpm < 150:
        return "normal"
    elif wpm < 180:
        return "fast"
    else:
        return "very_fast"

def analyze_speaking_rates(dataset_path: str, results_path: str) -> Dict[str, Any]:
    """Analyze speaking rates and correlate with STT accuracy."""
    
    # Load dataset and results
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create lookup for results by sample_id
    results_lookup = {}
    for result in results['individual_results']:
        results_lookup[result['sample_id']] = result
    
    # Analyze each sample
    analysis_data = []
    base_dir = Path(dataset_path).parent.parent
    
    for sample in dataset:
        sample_id = sample['id']
        
        # Get audio duration from raw audio file
        raw_audio_path = str(base_dir / sample['raw_audio'].lstrip('../'))
        duration = get_audio_duration(raw_audio_path)
        
        if sample_id in results_lookup:
            result = results_lookup[sample_id]
            ground_truth = result['ground_truth']
            
            # Calculate speaking rate
            speaking_rate = calculate_speaking_rate(ground_truth, duration)
            rate_category = categorize_speaking_rate(speaking_rate)
            
            # Get model accuracies
            model_accuracies = {}
            for model_name, model_result in result['models'].items():
                model_accuracies[model_name] = model_result['accuracy_percent']
            
            analysis_data.append({
                'sample_id': sample_id,
                'duration_seconds': duration,
                'word_count': len(ground_truth.split()),
                'speaking_rate_wpm': speaking_rate,
                'rate_category': rate_category,
                'model_accuracies': model_accuracies
            })
    
    return {
        'samples': analysis_data,
        'correlations': calculate_correlations(analysis_data),
        'rate_category_analysis': analyze_by_rate_category(analysis_data)
    }

def calculate_correlations(analysis_data: List[Dict]) -> Dict[str, Any]:
    """Calculate correlation between speaking rate and model accuracy."""
    correlations = {}
    
    # Get all model names
    model_names = set()
    for sample in analysis_data:
        model_names.update(sample['model_accuracies'].keys())
    
    for model_name in model_names:
        rates = []
        accuracies = []
        
        for sample in analysis_data:
            if model_name in sample['model_accuracies']:
                rates.append(sample['speaking_rate_wpm'])
                accuracies.append(sample['model_accuracies'][model_name])
        
        # Calculate Pearson correlation coefficient
        if len(rates) > 1:
            correlation = calculate_pearson_correlation(rates, accuracies)
            correlations[model_name] = {
                'correlation_coefficient': correlation,
                'interpretation': interpret_correlation(correlation),
                'sample_count': len(rates)
            }
    
    return correlations

def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] ** 2 for i in range(n))
    sum_y2 = sum(y[i] ** 2 for i in range(n))
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def interpret_correlation(correlation: float) -> str:
    """Interpret correlation coefficient strength."""
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        return "negligible"
    elif abs_corr < 0.3:
        return "weak"
    elif abs_corr < 0.5:
        return "moderate"
    elif abs_corr < 0.7:
        return "strong"
    else:
        return "very_strong"

def analyze_by_rate_category(analysis_data: List[Dict]) -> Dict[str, Any]:
    """Analyze performance by speaking rate category."""
    categories = {}
    
    for sample in analysis_data:
        category = sample['rate_category']
        if category not in categories:
            categories[category] = {
                'samples': [],
                'avg_rate': 0,
                'model_performance': {}
            }
        
        categories[category]['samples'].append(sample)
    
    # Calculate averages for each category
    for category, data in categories.items():
        samples = data['samples']
        data['sample_count'] = len(samples)
        data['avg_rate'] = sum(s['speaking_rate_wpm'] for s in samples) / len(samples)
        
        # Calculate average accuracy per model for this category
        model_names = set()
        for sample in samples:
            model_names.update(sample['model_accuracies'].keys())
        
        for model_name in model_names:
            accuracies = []
            for sample in samples:
                if model_name in sample['model_accuracies']:
                    accuracies.append(sample['model_accuracies'][model_name])
            
            if accuracies:
                data['model_performance'][model_name] = {
                    'avg_accuracy': sum(accuracies) / len(accuracies),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies),
                    'sample_count': len(accuracies)
                }
    
    return categories

def main():
    parser = argparse.ArgumentParser(description='Analyze speaking rates and correlate with STT accuracy')
    parser.add_argument('--dataset', required=True, help='Path to dataset JSON file')
    parser.add_argument('--results', required=True, help='Path to evaluation results JSON file')
    parser.add_argument('--output', help='Output file for analysis results (JSON)')
    
    args = parser.parse_args()
    
    print("Analyzing speaking rates and STT accuracy correlations...")
    analysis = analyze_speaking_rates(args.dataset, args.results)
    
    # Print summary
    print(f"\nSpeaking Rate Analysis Summary:")
    print(f"Total samples analyzed: {len(analysis['samples'])}")
    
    # Print rate distribution
    rate_counts = {}
    for sample in analysis['samples']:
        category = sample['rate_category']
        rate_counts[category] = rate_counts.get(category, 0) + 1
    
    print(f"\nSpeaking Rate Distribution:")
    for category, count in sorted(rate_counts.items()):
        print(f"  {category.title()}: {count} samples")
    
    # Print correlations
    print(f"\nCorrelation Analysis (Speaking Rate vs Accuracy):")
    for model_name, corr_data in analysis['correlations'].items():
        corr = corr_data['correlation_coefficient']
        interp = corr_data['interpretation']
        print(f"  {model_name}: r={corr:.3f} ({interp})")
    
    # Print category analysis
    print(f"\nPerformance by Speaking Rate Category:")
    for category, data in analysis['rate_category_analysis'].items():
        print(f"\n  {category.upper()} ({data['avg_rate']:.1f} WPM, {data['sample_count']} samples):")
        for model_name, perf in data['model_performance'].items():
            print(f"    {model_name}: {perf['avg_accuracy']:.1f}% avg accuracy")
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nDetailed analysis saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
