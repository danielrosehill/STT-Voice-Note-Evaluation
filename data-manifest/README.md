# Dataset Manifests

This directory contains JSON dataset files that map audio recordings to their corresponding ground truth transcripts for STT evaluation.

## Dataset Files

### Combined Datasets
- `english_dataset.json` - 16 English samples with both raw and denoised audio paths
- `multilingual_dataset.json` - 3 Hebrew-English samples with both raw and denoised audio paths

### Separated by Audio Processing
- `english_raw_dataset.json` - 16 English samples using raw audio only
- `english_denoised_dataset.json` - 16 English samples using denoised audio only
- `multilingual_raw_dataset.json` - 3 multilingual samples using raw audio only
- `multilingual_denoised_dataset.json` - 3 multilingual samples using denoised audio only

## Usage

### Loading Datasets in Python

```python
import json
import os

# Change to repository root
os.chdir('..')

# Load combined dataset (both raw and denoised paths)
with open('data-manifest/english_dataset.json') as f:
    english_data = json.load(f)

# Load specific audio processing type
with open('data-manifest/english_raw_dataset.json') as f:
    raw_english = json.load(f)

# Access files
for sample in english_data:
    print(f"ID: {sample['id']}")
    print(f"Raw audio: {sample['raw_audio']}")
    print(f"Denoised audio: {sample['denoised_audio']}")
    print(f"Ground truth: {sample['ground_truth']}")
```

### Path Structure

All paths in these JSON files are **relative to the repository root** using `../` notation since the manifests are in the `data-manifest/` subdirectory.

Example paths:
- Audio: `../audio/raw/english/01_email_dictation.wav`
- Transcripts: `../texts/01_email_dictation.txt`

## Evaluation Scenarios

These manifests support **4 evaluation scenarios**:

1. **Raw English** (16 samples) - Real-world phone audio quality
2. **Denoised English** (16 samples) - Preprocessed for optimal STT
3. **Raw Multilingual** (3 samples) - Hebrew-English code-switching
4. **Denoised Multilingual** (3 samples) - Processed bilingual content

## Schema

### Combined Dataset Format
```json
{
  "id": "sample_identifier",
  "raw_audio": "../audio/raw/[lang]/filename.wav",
  "denoised_audio": "../audio/denoised/[lang]/filename.wav", 
  "ground_truth": "../[texts|multilingual]/filename.txt"
}
```

### Single Audio Format
```json
{
  "id": "sample_identifier",
  "audio_file": "../audio/[raw|denoised]/[lang]/filename.wav",
  "ground_truth": "../[texts|multilingual]/filename.txt"
}
```
