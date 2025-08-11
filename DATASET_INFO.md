# Dataset Information

## Sample Size Analysis

**10 English samples + 2 multilingual samples** is a solid starting point for STT evaluation because:

### Statistical Considerations
- **Initial comparison**: 10 samples can reveal major differences between STT services (>10% accuracy gaps)
- **Technical vocabulary testing**: Sufficient to test how services handle Docker, GitHub, AI/ML terms
- **Content diversity**: Covers the main voice note categories you use
- **Iteration friendly**: Easy to expand if initial results show high variance

### When to Expand
Consider increasing to 20-50 samples if:
- WER differences between services are <5% (need more statistical power)
- You want to test specific edge cases (heavy accents, background noise)
- Planning to publish results or use for business decisions

### Content Categories Covered
1. **Technical discussions** (3 samples) - Docker, GitHub, AI workflows
2. **Project management** (2 samples) - Planning, updates, meetings  
3. **Personal organization** (2 samples) - Todo lists, scheduling
4. **Domain-specific** (2 samples) - Parenting questions, research notes
5. **Communication** (1 sample) - Email dictation
6. **Multilingual** (2 samples) - Hebrew-English code-switching

## Evaluation Methodology

### Primary Metric: Word Error Rate (WER)
- Industry standard for STT evaluation
- Formula: `(Substitutions + Deletions + Insertions) / Total_Reference_Words`
- Lower is better (0.0 = perfect, 1.0 = completely wrong)

### Secondary Metrics to Consider
- **Technical term accuracy**: How well does each service handle "Docker", "GitHub", "Kubernetes"?
- **Multilingual handling**: Can services detect and process Hebrew words correctly?
- **Disfluency handling**: How do services deal with "um", "uh", self-corrections?

## Expected Results

Based on typical STT performance:
- **Whisper**: Likely best for technical terms, good multilingual support
- **Deepgram**: Fast, good for real-time, may struggle with Hebrew
- **Google**: Strong overall, good multilingual, may over-correct casual speech
- **Azure**: Enterprise-focused, consistent but potentially conservative

## Next Steps

1. Record audio versions of these transcripts (or use existing recordings)
2. Process through each STT service
3. Run evaluation script
4. Analyze results and decide if expansion is needed
