---
license: apache-2.0
task_categories:
- text-generation
language:
- en
- he
pretty_name: Voice Note Speech To Text (STT) Evaluation Dataset
size_categories:
- n<1K
---

# STT Voice Note Evaluation

**Author:** Daniel Rosehill  
**Date Created:** August 11, 2025  
**Purpose:** Comparative evaluation of Speech-to-Text (STT) services for voice note transcription

## Overview

This dataset was created as part of ongoing work developing voice note transcription systems. It contains ground truth transcripts representing typical daily voice notes, recorded to evaluate and compare STT service accuracy across different content types.

**Speaker Profile:**
- Single speaker (Daniel Rosehill)
- Slight Irish accent
- Native English speaker living in Israel
- Frequent Hebrew-English code-switching in multilingual samples

**Content represents typical voice notes covering:**
- Technical discussions (Docker, GitHub, AI/ML)
- Project planning and management
- Personal tasks and scheduling
- Parenting questions and observations
- Research notes and troubleshooting
- Mixed English-Hebrew content

## Dataset Structure

```
├── texts/                    # English voice note transcripts (16 samples)
│   ├── 01_email_dictation.txt
│   ├── 02_project_planning.txt
│   ├── 03_todo_list.txt
│   ├── 04_meeting_notes.txt
│   ├── 05_parenting_question.txt
│   ├── 06_technical_troubleshooting.txt
│   ├── 07_blog_outline.txt
│   ├── 08_calendar_scheduling.txt
│   ├── 09_research_note.txt
│   ├── 10_project_update.txt
│   ├── 11_ai_prompt_creation.txt
│   ├── 12_agent_instructions.txt
│   ├── 13_pharmacy_pickup.txt
│   ├── 14_household_chores.txt
│   ├── 15_grocery_shopping.txt
│   └── 16_general_ai_prompt.txt
├── multilingual/             # Mixed English-Hebrew transcripts (3 samples)
│   ├── 01_teudat_zehut_pickup.txt
│   ├── 02_shabbat_preparations.txt
│   └── 03_shopping_list.txt
├── audio/                    # Audio recordings (WAV format)
│   ├── raw/                  # Original recordings
│   │   ├── english/          # 16 raw English voice notes
│   │   └── multilingual/     # 3 raw multilingual voice notes
│   └── denoised/             # Noise-reduced versions
│       ├── english/          # 16 denoised English voice notes
│       └── multilingual/     # 3 denoised multilingual voice notes
├── results/                  # STT API results (to be created)
├── scripts/                  # Utility scripts
│   ├── substitute_pii.sh     # PII anonymization script
│   └── denoise_audio.py      # Audio preprocessing script
├── data-manifest/            # Dataset manifests (JSON files)
│   ├── english_dataset.json       # Combined English samples (raw + denoised)
│   ├── multilingual_dataset.json  # Combined multilingual samples
│   ├── english_raw_dataset.json   # English raw audio only
│   ├── english_denoised_dataset.json # English denoised audio only
│   ├── multilingual_raw_dataset.json # Multilingual raw audio only
│   ├── multilingual_denoised_dataset.json # Multilingual denoised audio only
│   └── README.md             # Dataset manifest documentation
└── evaluate_stt.py          # Evaluation script for calculating WER
└── dataset.json              # Structured dataset metadata
```

## Content Characteristics

The voice notes in this dataset reflect natural speech patterns including:
- Stream-of-consciousness style
- Technical jargon mixed with casual language
- Self-corrections and hesitations
- Context switching between topics
- Intentional pauses to test hallucination handling
- Bilingual code-switching (English-Hebrew for immigrant usage patterns)

### Multilingual Content
The Hebrew-English samples represent common immigrant speech patterns where Hebrew words are naturally integrated into English conversation. This tests STT services' ability to handle:
- Administrative terms (teudat zehut, misrad hapnim)
- Religious/cultural terms (Shabbat, kiddush, nerot)
- Food and everyday items (lechem, yerakot, chamusim)
- Expectation of transliterated Hebrew in English transcripts

## Recording Conditions

**Environment**: Home office, quiet conditions (non-challenging acoustic environment)
**Device**: OnePlus phone (consumer-grade audio quality, mimicking real-world usage)
**Format**: WAV (lossless, optimal for STT evaluation)
**Preprocessing**: Both raw and denoised versions provided
**Limitations**: Recordings lack the audio background variation present in real-world voice note usage

## Audio Preprocessing

The dataset includes both **raw** and **denoised** versions of all recordings:

**Raw Audio:**
- Original recordings as captured by OnePlus phone
- Natural background noise and audio artifacts
- Tests STT robustness to real-world conditions

**Denoised Audio:**
- Processed using noise reduction algorithms
- Cleaner signal for optimal STT performance
- Tests impact of audio preprocessing on accuracy

This dual approach enables evaluation of:
1. **Raw performance** - How services handle unprocessed voice notes
2. **Preprocessing benefits** - Improvement gained from noise reduction
3. **Service sensitivity** - Which STT services are most affected by audio quality
4. **Cost-benefit analysis** - Whether audio preprocessing investment is worthwhile


## Usage

1. Use the ground truth transcripts in `texts/` and `multilingual/` as reference
2. Process the same audio through different STT APIs
3. Compare results using word error rate (WER) and other metrics
4. Store API results in `results/` directory

## Usage & Evaluation

This dataset is designed for:
1. **STT Service Comparison** - Evaluate accuracy across different providers (OpenAI Whisper, Deepgram, Google, Azure, etc.)
2. **Accent Impact Assessment** - Test how Irish accent affects transcription accuracy
3. **Multilingual Capability Testing** - Assess Hebrew-English code-switching handling
4. **Content Type Analysis** - Compare performance across technical vs. everyday language
5. **Pause/Silence Handling** - Evaluate hallucination tendencies during speech pauses

The dataset provides ground truth for calculating Word Error Rate (WER) and other accuracy metrics across different STT services to identify optimal solutions for voice note transcription systems.
