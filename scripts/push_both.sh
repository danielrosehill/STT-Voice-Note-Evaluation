#!/bin/bash
# Push to both GitHub and Hugging Face simultaneously

echo "🚀 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ GitHub push successful"
    echo "🤗 Pushing to Hugging Face..."
    
    # Try regular push first
    git push huggingface main
    
    if [ $? -eq 0 ]; then
        echo "✅ Hugging Face push successful"
        echo "🎉 Both remotes updated successfully!"
    else
        echo "⚠️  Regular push failed, trying with LFS..."
        git lfs push huggingface main
        
        if [ $? -eq 0 ]; then
            echo "✅ Hugging Face LFS push successful"
            echo "🎉 Both remotes updated successfully!"
        else
            echo "❌ Hugging Face push failed - may need manual sync"
            echo "💡 Try: huggingface-cli upload danielrosehill/Voice-Note-STT-Eval-Dataset . --repo-type=dataset"
            exit 1
        fi
    fi
else
    echo "❌ GitHub push failed"
    exit 1
fi
