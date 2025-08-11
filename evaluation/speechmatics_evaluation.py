#!/usr/bin/env python3
"""
Speechmatics STT Evaluation

Evaluates Speechmatics Nova-2 model using the same framework as OpenAI evaluation.
Saves transcriptions in organized text files and generates comparison-ready results.

Usage:
    python speechmatics_evaluation.py --dataset ../data-manifest/english_dataset.json --output ../results/speechmatics_evaluation_raw.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests
from datetime import datetime
import difflib

class SpeechmaticsEvaluator:
    def __init__(self, api_key: str, transcriptions_base_dir: str = "transcriptions"):
        self.api_key = api_key
        self.base_url = "https://asr.api.speechmatics.com/v2"
        self.model = "nova-2"
        self.vendor = "speechmatics"
        self.transcriptions_dir = Path(transcriptions_base_dir)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
    def get_transcription_path(self, sample_id: str, audio_type: str = "raw") -> Path:
        model_dir = self.transcriptions_dir / self.vendor / self.model / audio_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{sample_id}.txt"
    
    def has_cached_transcription(self, sample_id: str, audio_type: str = "raw") -> bool:
        return self.get_transcription_path(sample_id, audio_type).exists()
    
    def load_cached_transcription(self, sample_id: str, audio_type: str = "raw") -> Optional[str]:
        transcription_path = self.get_transcription_path(sample_id, audio_type)
        if transcription_path.exists():
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not load cached transcription: {e}")
        return None
    
    def save_transcription(self, sample_id: str, transcription: str, audio_type: str = "raw") -> None:
        transcription_path = self.get_transcription_path(sample_id, audio_type)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        operations = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                operations += max(i2 - i1, j2 - j1)
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else float('inf')
        
        return operations / len(ref_words)

    def transcribe_with_speechmatics(self, audio_path: str) -> str:
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            config = {
                "type": "transcription",
                "transcription_config": {
                    "language": "en",
                    "operating_point": "enhanced"
                }
            }
            
            # Submit job
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'data_file': audio_file,
                    'config': (None, json.dumps(config), 'application/json')
                }
                
                response = requests.post(f"{self.base_url}/jobs", headers=headers, files=files)
            
            if response.status_code != 201:
                print(f"Error submitting job: {response.status_code} - {response.text}")
                return ""
            
            job_id = response.json()['id']
            print(f"    Job ID: {job_id}")
            
            # Poll for completion
            while True:
                response = requests.get(f"{self.base_url}/jobs/{job_id}", headers=headers)
                
                if response.status_code != 200:
                    print(f"Error checking status: {response.status_code}")
                    return ""
                
                job_status = response.json()['job']['status']
                
                if job_status == 'done':
                    break
                elif job_status == 'rejected':
                    print(f"Job rejected: {response.json()}")
                    return ""
                
                time.sleep(2)
            
            # Get transcript
            response = requests.get(
                f"{self.base_url}/jobs/{job_id}/transcript",
                headers=headers,
                params={'format': 'txt'}
            )
            
            if response.status_code != 200:
                print(f"Error getting transcript: {response.status_code}")
                return ""
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error with Speechmatics: {e}")
            return ""

    def load_ground_truth(self, ground_truth_path: str) -> str:
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return ""

    def evaluate_sample(self, sample: Dict[str, str], use_denoised: bool = True, 
                       force_retranscribe: bool = False) -> Dict[str, Any]:
        sample_id = sample['id']
        audio_path = sample['denoised_audio'] if use_denoised else sample['raw_audio']
        ground_truth_path = sample['ground_truth']
        audio_type = 'denoised' if use_denoised else 'raw'
        
        # Convert relative paths to absolute paths
        base_dir = Path(__file__).parent.parent
        audio_path = str(base_dir / audio_path.lstrip('../'))
        ground_truth_path = str(base_dir / ground_truth_path.lstrip('../'))
        
        print(f"\nEvaluating sample: {sample_id}")
        print(f"Audio file: {audio_path}")
        
        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_path)
        if not ground_truth:
            return None
            
        # Check cache first
        if not force_retranscribe and self.has_cached_transcription(sample_id, audio_type):
            transcription = self.load_cached_transcription(sample_id, audio_type)
            if transcription:
                print(f"  Using cached transcription")
                processing_time = 0.0
            else:
                transcription = ""
                processing_time = 0.0
        else:
            # Perform transcription
            print(f"  Transcribing with Speechmatics Nova-2...")
            start_time = time.time()
            transcription = self.transcribe_with_speechmatics(audio_path)
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Save transcription
            if transcription:
                self.save_transcription(sample_id, transcription, audio_type)
                print(f"  Completed in {processing_time:.2f}s")
        
        if transcription:
            wer = self.calculate_wer(ground_truth, transcription)
            accuracy = (1 - wer) * 100
            
            result = {
                'sample_id': sample_id,
                'audio_type': audio_type,
                'ground_truth': ground_truth,
                'ground_truth_word_count': len(ground_truth.split()),
                'transcription': transcription,
                'word_count': len(transcription.split()),
                'wer': wer,
                'accuracy_percent': accuracy,
                'processing_time_seconds': processing_time
            }
            
            print(f"  Accuracy: {accuracy:.1f}% (WER: {wer:.3f})")
            return result
        else:
            print(f"  Failed to transcribe")
            return {
                'sample_id': sample_id,
                'audio_type': audio_type,
                'ground_truth': ground_truth,
                'ground_truth_word_count': len(ground_truth.split()),
                'transcription': "",
                'word_count': 0,
                'wer': float('inf'),
                'accuracy_percent': 0.0,
                'processing_time_seconds': processing_time,
                'error': True
            }

    def run_evaluation(self, dataset_path: str, use_denoised: bool = True, 
                      force_retranscribe: bool = False) -> Dict[str, Any]:
        print(f"Starting Speechmatics Nova-2 STT Evaluation")
        print(f"Dataset: {dataset_path}")
        print(f"Audio type: {'denoised' if use_denoised else 'raw'}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        evaluation_results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'audio_type': 'denoised' if use_denoised else 'raw',
                'total_samples': len(dataset),
                'model': 'speechmatics-nova-2',
                'transcriptions_directory': str(self.transcriptions_dir)
            },
            'individual_results': []
        }
        
        # Process each sample
        for i, sample in enumerate(dataset, 1):
            print(f"\n{'='*60}")
            print(f"Processing sample {i}/{len(dataset)}")
            
            result = self.evaluate_sample(sample, use_denoised, force_retranscribe)
            if result:
                evaluation_results['individual_results'].append(result)
            
            # Add delay between samples to be respectful to API
            if i < len(dataset):
                time.sleep(1)
        
        # Calculate summary statistics
        successful_results = [r for r in evaluation_results['individual_results'] if not r.get('error', False)]
        
        if successful_results:
            accuracies = [r['accuracy_percent'] for r in successful_results]
            wers = [r['wer'] for r in successful_results]
            times = [r['processing_time_seconds'] for r in successful_results if r['processing_time_seconds'] > 0]
            
            evaluation_results['summary_statistics'] = {
                'successful_transcriptions': len(successful_results),
                'total_samples': len(evaluation_results['individual_results']),
                'success_rate_percent': (len(successful_results) / len(evaluation_results['individual_results'])) * 100,
                'average_accuracy_percent': sum(accuracies) / len(accuracies),
                'average_wer': sum(wers) / len(wers),
                'min_accuracy_percent': min(accuracies),
                'max_accuracy_percent': max(accuracies),
                'average_processing_time_seconds': sum(times) / len(times) if times else 0,
                'total_processing_time_seconds': sum(times) if times else 0
            }
        
        return evaluation_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Speechmatics Nova-2 STT model')
    parser.add_argument('--dataset', required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', required=True, help='Output file for results (JSON)')
    parser.add_argument('--raw-audio', action='store_true', help='Use raw audio instead of denoised')
    parser.add_argument('--force-retranscribe', action='store_true', help='Force retranscription even if cached')
    parser.add_argument('--api-key', help='Speechmatics API key (or set SPEECHMATICS_API env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('SPEECHMATICS_API')
    if not api_key:
        print("Error: Speechmatics API key required. Set SPEECHMATICS_API environment variable or use --api-key")
        return 1
    
    # Initialize evaluator
    evaluator = SpeechmaticsEvaluator(api_key)
    
    # Run evaluation
    use_denoised = not args.raw_audio
    results = evaluator.run_evaluation(args.dataset, use_denoised, args.force_retranscribe)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SPEECHMATICS EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    if 'summary_statistics' in results and results['summary_statistics']:
        stats = results['summary_statistics']
        print(f"\nSUMMARY RESULTS:")
        print(f"Success Rate: {stats['success_rate_percent']:.1f}%")
        print(f"Average Accuracy: {stats['average_accuracy_percent']:.1f}%")
        print(f"Average WER: {stats['average_wer']:.3f}")
        print(f"Accuracy Range: {stats['min_accuracy_percent']:.1f}% - {stats['max_accuracy_percent']:.1f}%")
        print(f"Avg Processing Time: {stats['average_processing_time_seconds']:.2f}s")
    
    print(f"\nTranscriptions saved in: transcriptions/speechmatics/nova-2/")
    print(f"Detailed results saved to: {args.output}")
    return 0

if __name__ == "__main__":
    exit(main())
