#!/usr/bin/env python3
"""
OpenAI Whisper vs GPT-4o Transcription Models Comparison

This script evaluates three OpenAI transcription models head-to-head:
- Legacy Whisper (whisper-1)
- GPT-4o Audio (gpt-4o-audio-preview) 
- GPT-4o Mini Audio (gpt-4o-mini-audio-preview)

Usage:
    python openai_whisper_comparison.py --dataset ../data-manifest/english_dataset.json --output results/openai_comparison.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import openai
from datetime import datetime
import difflib

class OpenAITranscriptionEvaluator:
    def __init__(self, api_key: str):
        """Initialize the evaluator with OpenAI API key."""
        self.client = openai.OpenAI(api_key=api_key)
        self.models = {
            'whisper-legacy': 'whisper-1',
            'gpt-4o-transcribe': 'gpt-4o-audio-preview',
            'gpt-4o-mini-transcribe': 'gpt-4o-mini-audio-preview'
        }
        
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
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

    def transcribe_with_whisper_legacy(self, audio_path: str) -> str:
        """Transcribe audio using legacy Whisper model."""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            print(f"Error with Whisper legacy on {audio_path}: {e}")
            return ""

    def transcribe_with_gpt4o(self, audio_path: str, model_name: str) -> str:
        """Transcribe audio using GPT-4o audio models."""
        try:
            import base64
            
            with open(audio_path, "rb") as audio_file:
                # Read and encode the audio file
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
            # For GPT-4o models, we need to use the chat completions API
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please transcribe this audio file accurately. Return only the transcription text without any additional commentary."
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
                temperature=0
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with {model_name} on {audio_path}: {e}")
            return ""

    def load_ground_truth(self, ground_truth_path: str) -> str:
        """Load ground truth transcript from file."""
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading ground truth from {ground_truth_path}: {e}")
            return ""

    def evaluate_sample(self, sample: Dict[str, str], use_denoised: bool = True) -> Dict[str, Any]:
        """Evaluate a single audio sample against all three models."""
        sample_id = sample['id']
        audio_path = sample['denoised_audio'] if use_denoised else sample['raw_audio']
        ground_truth_path = sample['ground_truth']
        
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
            
        results = {
            'sample_id': sample_id,
            'audio_type': 'denoised' if use_denoised else 'raw',
            'ground_truth': ground_truth,
            'ground_truth_word_count': len(ground_truth.split()),
            'models': {}
        }
        
        # Test each model
        for model_key, model_name in self.models.items():
            print(f"  Testing {model_key} ({model_name})...")
            
            start_time = time.time()
            
            if model_key == 'whisper-legacy':
                transcription = self.transcribe_with_whisper_legacy(audio_path)
            else:
                transcription = self.transcribe_with_gpt4o(audio_path, model_name)
            
            end_time = time.time()
            
            if transcription:
                wer = self.calculate_wer(ground_truth, transcription)
                accuracy = (1 - wer) * 100
                
                results['models'][model_key] = {
                    'model_name': model_name,
                    'transcription': transcription,
                    'word_count': len(transcription.split()),
                    'wer': wer,
                    'accuracy_percent': accuracy,
                    'processing_time_seconds': end_time - start_time
                }
                
                print(f"    Accuracy: {accuracy:.1f}% (WER: {wer:.3f})")
                print(f"    Processing time: {end_time - start_time:.2f}s")
            else:
                results['models'][model_key] = {
                    'model_name': model_name,
                    'transcription': "",
                    'word_count': 0,
                    'wer': float('inf'),
                    'accuracy_percent': 0.0,
                    'processing_time_seconds': end_time - start_time,
                    'error': True
                }
                print(f"    Failed to transcribe")
            
            # Add delay between API calls to avoid rate limiting
            time.sleep(1)
        
        return results

    def run_evaluation(self, dataset_path: str, use_denoised: bool = True) -> Dict[str, Any]:
        """Run complete evaluation on the dataset."""
        print(f"Starting OpenAI Whisper vs GPT-4o Transcription Evaluation")
        print(f"Dataset: {dataset_path}")
        print(f"Audio type: {'denoised' if use_denoised else 'raw'}")
        print(f"Models: {list(self.models.keys())}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        evaluation_results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'audio_type': 'denoised' if use_denoised else 'raw',
                'total_samples': len(dataset),
                'models_tested': list(self.models.keys())
            },
            'individual_results': [],
            'summary_statistics': {}
        }
        
        # Process each sample
        successful_evaluations = 0
        for i, sample in enumerate(dataset, 1):
            print(f"\n{'='*50}")
            print(f"Processing sample {i}/{len(dataset)}")
            
            result = self.evaluate_sample(sample, use_denoised)
            if result:
                evaluation_results['individual_results'].append(result)
                successful_evaluations += 1
        
        # Calculate summary statistics
        if successful_evaluations > 0:
            summary = {}
            for model_key in self.models.keys():
                model_results = []
                total_time = 0
                successful_transcriptions = 0
                
                for result in evaluation_results['individual_results']:
                    if model_key in result['models'] and not result['models'][model_key].get('error', False):
                        model_results.append(result['models'][model_key])
                        total_time += result['models'][model_key]['processing_time_seconds']
                        successful_transcriptions += 1
                
                if model_results:
                    accuracies = [r['accuracy_percent'] for r in model_results]
                    wers = [r['wer'] for r in model_results]
                    
                    summary[model_key] = {
                        'model_name': self.models[model_key],
                        'successful_transcriptions': successful_transcriptions,
                        'total_samples': len(evaluation_results['individual_results']),
                        'success_rate_percent': (successful_transcriptions / len(evaluation_results['individual_results'])) * 100,
                        'average_accuracy_percent': sum(accuracies) / len(accuracies),
                        'average_wer': sum(wers) / len(wers),
                        'min_accuracy_percent': min(accuracies),
                        'max_accuracy_percent': max(accuracies),
                        'average_processing_time_seconds': total_time / successful_transcriptions if successful_transcriptions > 0 else 0,
                        'total_processing_time_seconds': total_time
                    }
            
            evaluation_results['summary_statistics'] = summary
        
        return evaluation_results

def main():
    parser = argparse.ArgumentParser(description='Compare OpenAI Whisper models on voice note transcription')
    parser.add_argument('--dataset', required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', required=True, help='Output file for results (JSON)')
    parser.add_argument('--raw-audio', action='store_true', help='Use raw audio instead of denoised')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_KEY environment variable or use --api-key")
        return 1
    
    # Initialize evaluator
    evaluator = OpenAITranscriptionEvaluator(api_key)
    
    # Run evaluation
    use_denoised = not args.raw_audio
    results = evaluator.run_evaluation(args.dataset, use_denoised)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    if 'summary_statistics' in results and results['summary_statistics']:
        print("\nSUMMARY RESULTS:")
        print("-" * 40)
        
        for model_key, stats in results['summary_statistics'].items():
            print(f"\n{model_key.upper()} ({stats['model_name']}):")
            print(f"  Success Rate: {stats['success_rate_percent']:.1f}%")
            print(f"  Average Accuracy: {stats['average_accuracy_percent']:.1f}%")
            print(f"  Average WER: {stats['average_wer']:.3f}")
            print(f"  Accuracy Range: {stats['min_accuracy_percent']:.1f}% - {stats['max_accuracy_percent']:.1f}%")
            print(f"  Avg Processing Time: {stats['average_processing_time_seconds']:.2f}s")
        
        # Determine winner
        best_model = max(results['summary_statistics'].items(), 
                        key=lambda x: x[1]['average_accuracy_percent'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0].upper()}")
        print(f"   Average Accuracy: {best_model[1]['average_accuracy_percent']:.1f}%")
    
    print(f"\nDetailed results saved to: {args.output}")
    return 0

if __name__ == "__main__":
    exit(main())
