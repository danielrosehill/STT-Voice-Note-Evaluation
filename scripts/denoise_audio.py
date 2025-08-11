#!/usr/bin/env python3
"""
Audio Denoising Script for STT Evaluation Dataset

This script processes raw audio files and creates denoised versions
for comparative STT evaluation.

Requirements:
    pip install noisereduce librosa soundfile

Usage:
    python denoise_audio.py --input audio/raw --output audio/denoised
"""

import argparse
import os
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path

def denoise_audio_file(input_path, output_path, sr=22050):
    """
    Apply noise reduction to an audio file.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save denoised audio
        sr: Sample rate for processing
    """
    try:
        # Load audio file
        audio, original_sr = librosa.load(input_path, sr=sr)
        
        # Apply noise reduction
        # Use first 1 second as noise sample for stationary noise reduction
        noise_sample = audio[:sr]  # First second
        denoised_audio = nr.reduce_noise(
            y=audio, 
            sr=sr,
            y_noise=noise_sample,
            stationary=True,
            prop_decrease=0.8
        )
        
        # Save denoised audio
        sf.write(output_path, denoised_audio, sr)
        print(f"✓ Processed: {input_path.name} -> {output_path.name}")
        
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")

def process_directory(input_dir, output_dir):
    """Process all WAV files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all WAV files
    wav_files = list(input_path.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Processing {len(wav_files)} files from {input_dir}")
    
    for wav_file in wav_files:
        output_file = output_path / wav_file.name
        denoise_audio_file(wav_file, output_file)

def main():
    parser = argparse.ArgumentParser(description='Denoise audio files for STT evaluation')
    parser.add_argument('--input', default='audio/raw', 
                       help='Input directory containing raw audio files')
    parser.add_argument('--output', default='audio/denoised',
                       help='Output directory for denoised audio files')
    
    args = parser.parse_args()
    
    # Process English files
    english_input = Path(args.input) / 'english'
    english_output = Path(args.output) / 'english'
    
    if english_input.exists():
        print("Processing English audio files...")
        process_directory(english_input, english_output)
    
    # Process Multilingual files
    multilingual_input = Path(args.input) / 'multilingual'
    multilingual_output = Path(args.output) / 'multilingual'
    
    if multilingual_input.exists():
        print("\nProcessing Multilingual audio files...")
        process_directory(multilingual_input, multilingual_output)
    
    print("\nDenoising complete!")
    print(f"Raw files: {args.input}")
    print(f"Denoised files: {args.output}")

if __name__ == "__main__":
    main()
