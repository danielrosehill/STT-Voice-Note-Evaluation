#!/bin/bash
# Deploy STT Voice Note Evaluation Dataset to Hugging Face
# Run this periodically to sync the latest changes from GitHub to HF

set -e  # Exit on any error

echo "🤗 Deploying to Hugging Face Dataset Repository..."
echo "📊 Repository: danielrosehill/Voice-Note-STT-Eval-Dataset"
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "texts" ]; then
    echo "❌ Error: Must be run from the STT-Voice-Note-Evaluation repository root"
    exit 1
fi

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ Error: huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
    exit 1
fi

# Ensure we have the latest from GitHub
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Deploy to Hugging Face using the CLI
echo "🚀 Uploading dataset to Hugging Face..."
huggingface-cli upload danielrosehill/Voice-Note-STT-Eval-Dataset . \
    --repo-type=dataset \
    --commit-message="Sync from GitHub: $(git log -1 --pretty=format:'%s')" \
    --commit-description="Automated deployment from GitHub repository"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully deployed to Hugging Face!"
    echo "🔗 Dataset available at: https://huggingface.co/datasets/danielrosehill/Voice-Note-STT-Eval-Dataset"
    echo ""
    echo "📈 Dataset includes:"
    echo "   • $(find texts/ -name '*.txt' | wc -l) English voice note transcripts"
    echo "   • $(find multilingual/ -name '*.txt' | wc -l) multilingual samples"
    echo "   • $(find audio/ -name '*.wav' | wc -l) audio files (raw + denoised)"
    echo "   • $(find data-manifest/ -name '*.json' | wc -l) JSON dataset manifests"
else
    echo "❌ Deployment failed!"
    echo "💡 Try running with --verbose flag for more details"
    exit 1
fi
