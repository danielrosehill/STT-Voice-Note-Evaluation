# Audio Recordings

This folder contains voice note recordings in WAV format, organized into raw and denoised versions for comprehensive STT evaluation.

## Structure

```
audio/
├── raw/                      # Original recordings
│   ├── english/              # 16 English voice notes (.wav)
│   │   ├── 01_email_dictation.wav
│   │   ├── 02_project_planning.wav
│   │   ├── ...
│   │   └── 16_general_ai_prompt.wav
│   └── multilingual/         # 3 Hebrew-English voice notes (.wav)
│       ├── 01_teudat_zehut_pickup.wav
│       ├── 02_shabbat_preparations.wav
│       └── 03_shopping_list.wav
└── denoised/                 # Noise-reduced versions
    ├── english/              # 16 processed English files
    └── multilingual/         # 3 processed multilingual files
```

## Audio Specifications

- **Format**: WAV (lossless, optimal for STT evaluation)
- **Source**: OnePlus phone recordings (consumer-grade quality)
- **Environment**: Home office, quiet conditions
- **Duration**: 1-2 minutes per sample
- **Processing**: Both raw and denoised versions available

## Denoising Process

To create denoised versions, use the provided script:

```bash
# Install dependencies
pip install noisereduce librosa soundfile

# Run denoising script
python scripts/denoise_audio.py --input audio/raw --output audio/denoised
```

The denoising script:
- Uses the first second of each recording as noise sample
- Applies stationary noise reduction
- Preserves speech quality while reducing background noise
- Maintains WAV format and sample rate

## Evaluation Usage

These audio files enable **4 evaluation scenarios**:
1. **Raw English** (16 samples) - Original quality
2. **Denoised English** (16 samples) - Noise-reduced
3. **Raw Multilingual** (3 samples) - Original Hebrew-English
4. **Denoised Multilingual** (3 samples) - Processed Hebrew-English

Compare STT accuracy across raw vs. denoised to assess preprocessing benefits.
