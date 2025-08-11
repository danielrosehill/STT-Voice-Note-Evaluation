#!/usr/bin/env python3
"""
OpenAI Comprehensive STT Evaluation

This script evaluates OpenAI transcription models and saves results in both:
1. Human-readable text files organized by vendor/model
2. Runtime metadata for programmatic analysis
3. Comprehensive evaluation results with caching to avoid re-running API calls

Directory structure:
transcriptions/
├── openai/
│   ├── whisper-1/
│   │   ├── denoised/
│   │   │   ├── 01_email_dictation.txt
│   │   │   ├── 02_project_planning.txt
│   │   │   └── ...
│   │   └── runtime_metadata.json
│   ├── gpt-4o-audio-preview/
│   └── gpt-4o-mini-audio-preview/

Usage:
    python openai_comprehensive_evaluation.py --dataset ../data-manifest/english_dataset.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import openai
from datetime import datetime
import difflib

class ComprehensiveSTTEvaluator:
    def __init__(self, api_key: str, transcriptions_base_dir: str = "transcriptions"):
        """Initialize the evaluator with OpenAI API key and transcription directory."""
        self.client = openai.OpenAI(api_key=api_key)
        self.models = {
            'whisper-1': 'whisper-1',
            'gpt-4o-audio-preview': 'gpt-4o-audio-preview', 
            'gpt-4o-mini-audio-preview': 'gpt-4o-mini-audio-preview'
        }
        self.vendor = "openai"
        self.transcriptions_dir = Path(transcriptions_base_dir)
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
    def get_transcription_path(self, model_name: str, sample_id: str, audio_type: str = "denoised") -> Path:
        """Get the path for saving a transcription text file."""
        model_dir = self.transcriptions_dir / self.vendor / model_name / audio_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{sample_id}.txt"
    
    def get_metadata_path(self, model_name: str) -> Path:
        """Get the path for saving runtime metadata."""
        model_dir = self.transcriptions_dir / self.vendor / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / "runtime_metadata.json"
    
    def load_existing_metadata(self, model_name: str) -> Dict[str, Any]:
        """Load existing runtime metadata if it exists."""
        metadata_path = self.get_metadata_path(model_name)
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing metadata for {model_name}: {e}")
        
        return {
            "model": model_name,
            "vendor": self.vendor,
            "transcriptions": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_metadata(self, model_name: str, metadata: Dict[str, Any]) -> None:
        """Save runtime metadata."""
        metadata_path = self.get_metadata_path(model_name)
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def has_cached_transcription(self, model_name: str, sample_id: str, audio_type: str = "denoised") -> bool:
        """Check if transcription already exists."""
        transcription_path = self.get_transcription_path(model_name, sample_id, audio_type)
        return transcription_path.exists()
    
    def load_cached_transcription(self, model_name: str, sample_id: str, audio_type: str = "denoised") -> Optional[str]:
        """Load existing transcription if it exists."""
        transcription_path = self.get_transcription_path(model_name, sample_id, audio_type)
        if transcription_path.exists():
            try:
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not load cached transcription from {transcription_path}: {e}")
        return None
    
    def save_transcription(self, model_name: str, sample_id: str, transcription: str, 
                          processing_time: float, audio_type: str = "denoised") -> None:
        """Save transcription to text file and update metadata."""
        # Save transcription text file
        transcription_path = self.get_transcription_path(model_name, sample_id, audio_type)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Update metadata
        metadata = self.load_existing_metadata(model_name)
        metadata["transcriptions"][f"{sample_id}_{audio_type}"] = {
            "sample_id": sample_id,
            "audio_type": audio_type,
            "transcription_file": str(transcription_path.relative_to(self.transcriptions_dir)),
            "processing_time_seconds": processing_time,
            "word_count": len(transcription.split()),
            "transcribed_at": datetime.now().isoformat()
        }
        self.save_metadata(model_name, metadata)
    
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

    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe audio using Whisper model."""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            print(f"Error with Whisper on {audio_path}: {e}")
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

    def transcribe_sample(self, model_name: str, audio_path: str, sample_id: str, 
                         audio_type: str = "denoised", force_retranscribe: bool = False) -> Tuple[str, float]:
        """Transcribe a single sample, using cache if available."""
        
        # Check cache first unless forced to retranscribe
        if not force_retranscribe and self.has_cached_transcription(model_name, sample_id, audio_type):
            cached_transcription = self.load_cached_transcription(model_name, sample_id, audio_type)
            if cached_transcription:
                print(f"    Using cached transcription")
                return cached_transcription, 0.0
        
        # Perform transcription
        print(f"    Transcribing with {model_name}...")
        start_time = time.time()
        
        if model_name == "whisper-1":
            transcription = self.transcribe_with_whisper(audio_path)
        else:
            transcription = self.transcribe_with_gpt4o(audio_path, model_name)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save transcription and metadata
        if transcription:
            self.save_transcription(model_name, sample_id, transcription, processing_time, audio_type)
            print(f"    Completed in {processing_time:.2f}s")
        else:
            print(f"    Failed to transcribe")
        
        return transcription, processing_time

    def load_ground_truth(self, ground_truth_path: str) -> str:
        """Load ground truth transcript from file."""
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading ground truth from {ground_truth_path}: {e}")
            return ""

    def evaluate_sample(self, sample: Dict[str, str], use_denoised: bool = True, 
                       force_retranscribe: bool = False) -> Dict[str, Any]:
        """Evaluate a single audio sample against all models."""
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
        print(f"Audio type: {audio_type}")
        
        # Load ground truth
        ground_truth = self.load_ground_truth(ground_truth_path)
        if not ground_truth:
            return None
            
        results = {
            'sample_id': sample_id,
            'audio_type': audio_type,
            'ground_truth': ground_truth,
            'ground_truth_word_count': len(ground_truth.split()),
            'models': {}
        }
        
        # Test each model
        for model_name in self.models.keys():
            print(f"  Testing {model_name}...")
            
            transcription, processing_time = self.transcribe_sample(
                model_name, audio_path, sample_id, audio_type, force_retranscribe
            )
            
            if transcription:
                wer = self.calculate_wer(ground_truth, transcription)
                accuracy = (1 - wer) * 100
                
                results['models'][model_name] = {
                    'transcription': transcription,
                    'word_count': len(transcription.split()),
                    'wer': wer,
                    'accuracy_percent': accuracy,
                    'processing_time_seconds': processing_time
                }
                
                print(f"    Accuracy: {accuracy:.1f}% (WER: {wer:.3f})")
                if processing_time > 0:
                    print(f"    Processing time: {processing_time:.2f}s")
            else:
                results['models'][model_name] = {
                    'transcription': "",
                    'word_count': 0,
                    'wer': float('inf'),
                    'accuracy_percent': 0.0,
                    'processing_time_seconds': processing_time,
                    'error': True
                }
                print(f"    Failed to transcribe")
            
            # Add delay between API calls to avoid rate limiting
            if processing_time > 0:  # Only delay if we actually made an API call
                time.sleep(1)
        
        return results

    def run_evaluation(self, dataset_path: str, use_denoised: bool = True, 
                      force_retranscribe: bool = False) -> Dict[str, Any]:
        """Run complete evaluation on the dataset."""
        print(f"Starting OpenAI Comprehensive STT Evaluation")
        print(f"Dataset: {dataset_path}")
        print(f"Audio type: {'denoised' if use_denoised else 'raw'}")
        print(f"Models: {list(self.models.keys())}")
        print(f"Transcriptions directory: {self.transcriptions_dir}")
        print(f"Force retranscribe: {force_retranscribe}")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        evaluation_results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': dataset_path,
                'audio_type': 'denoised' if use_denoised else 'raw',
                'total_samples': len(dataset),
                'models_tested': list(self.models.keys()),
                'transcriptions_directory': str(self.transcriptions_dir),
                'force_retranscribe': force_retranscribe
            },
            'individual_results': [],
            'summary_statistics': {}
        }
        
        # Process each sample
        successful_evaluations = 0
        for i, sample in enumerate(dataset, 1):
            print(f"\n{'='*60}")
            print(f"Processing sample {i}/{len(dataset)}")
            
            result = self.evaluate_sample(sample, use_denoised, force_retranscribe)
            if result:
                evaluation_results['individual_results'].append(result)
                successful_evaluations += 1
        
        # Calculate summary statistics
        if successful_evaluations > 0:
            summary = {}
            for model_name in self.models.keys():
                model_results = []
                total_time = 0
                successful_transcriptions = 0
                
                for result in evaluation_results['individual_results']:
                    if model_name in result['models'] and not result['models'][model_name].get('error', False):
                        model_results.append(result['models'][model_name])
                        total_time += result['models'][model_name]['processing_time_seconds']
                        successful_transcriptions += 1
                
                if model_results:
                    accuracies = [r['accuracy_percent'] for r in model_results]
                    wers = [r['wer'] for r in model_results]
                    
                    summary[model_name] = {
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

    def print_cache_status(self) -> None:
        """Print status of cached transcriptions."""
        print(f"\nCached Transcriptions Status:")
        print(f"Base directory: {self.transcriptions_dir}")
        
        for model_name in self.models.keys():
            model_dir = self.transcriptions_dir / self.vendor / model_name
            if model_dir.exists():
                denoised_dir = model_dir / "denoised"
                raw_dir = model_dir / "raw"
                
                denoised_count = len(list(denoised_dir.glob("*.txt"))) if denoised_dir.exists() else 0
                raw_count = len(list(raw_dir.glob("*.txt"))) if raw_dir.exists() else 0
                
                print(f"  {model_name}: {denoised_count} denoised, {raw_count} raw transcriptions")
            else:
                print(f"  {model_name}: No cached transcriptions")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive OpenAI STT Evaluation with Caching')
    parser.add_argument('--dataset', required=True, help='Path to dataset JSON file')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--raw-audio', action='store_true', help='Use raw audio instead of denoised')
    parser.add_argument('--force-retranscribe', action='store_true', help='Force retranscription even if cached')
    parser.add_argument('--transcriptions-dir', default='transcriptions', help='Base directory for transcriptions')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_KEY env var)')
    parser.add_argument('--cache-status', action='store_true', help='Show cache status and exit')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_KEY')
    if not api_key and not args.cache_status:
        print("Error: OpenAI API key required. Set OPENAI_KEY environment variable or use --api-key")
        return 1
    
    # Initialize evaluator
    evaluator = ComprehensiveSTTEvaluator(api_key or "", args.transcriptions_dir)
    
    # Show cache status if requested
    if args.cache_status:
        evaluator.print_cache_status()
        return 0
    
    # Show current cache status
    evaluator.print_cache_status()
    
    # Run evaluation
    use_denoised = not args.raw_audio
    results = evaluator.run_evaluation(args.dataset, use_denoised, args.force_retranscribe)
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    if 'summary_statistics' in results and results['summary_statistics']:
        print("\nSUMMARY RESULTS:")
        print("-" * 40)
        
        for model_name, stats in results['summary_statistics'].items():
            print(f"\n{model_name.upper()}:")
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
    
    print(f"\nTranscriptions saved in: {evaluator.transcriptions_dir}")
    return 0

if __name__ == "__main__":
    exit(main())
