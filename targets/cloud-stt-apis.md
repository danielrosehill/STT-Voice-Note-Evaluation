# Cloud STT Evaluation Targets

## Models

| Provider     | Model                   | Why It Might Be Useful                                             | Approx. Cost\*                                  |
| ------------ | ----------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| Google Cloud | Speech-to-Text v2       | Mature API, good accuracy, phrase hints, multilingual              | ~~\$0.006–\$0.009 / 15 sec (~~\$1.44–\$2.16/hr) |
| AWS          | Transcribe Standard     | Easy S3 batch, good vocab customization, stable enterprise support | ~~\$0.0004/sec (~~\$1.44/hr)                    |
| Azure        | Speech to Text Standard | Good for enterprise integrations, solid with en-GB accents         | \~\$1/hr                                        |
| Deepgram     | Nova-2                  | Fast, competitive accuracy, strong punctuation                     | ~~\$0.004/min (~~\$0.24/hr)                     |
| AssemblyAI   | Best                    | Simple API, strong on noisy audio, extra AI features               | ~~\$0.008/min (~~\$0.48/hr)                     |
| Speechmatics | Any-to-Text Cloud       | Very good with accents, custom lexicons                            | ~~\$0.004/min (~~\$0.24/hr)                     |
| Rev AI       | Reverb                  | Pragmatic choice, affordable, human fallback option                | ~~\$0.0033–\$0.0066/min (~~\$0.20–\$0.40/hr)    |
| OpenAI       | whisper-1               | Open-source lineage, good accuracy, cost-effective                 | ~~\$0.006/min (~~\$0.36/hr)                     |
| OpenAI       | gpt-4o-transcribe       | Strong context handling, advanced formatting                       | ~~\$0.012/min (~~\$0.72/hr)                     |

 
---

## By Cost (Most To Least)

| Provider     | Model                   | Approx. Cost/hr |
| ------------ | ----------------------- | --------------- |
| Google Cloud | Speech-to-Text v2       | \~\$1.44–\$2.16 |
| AWS          | Transcribe Standard     | \~\$1.44        |
| Azure        | Speech to Text Standard | \~\$1.00        |
| OpenAI       | gpt-4o-transcribe       | \~\$0.72        |
| AssemblyAI   | Best                    | \~\$0.48        |
| OpenAI       | whisper-1               | \~\$0.36        |
| Rev AI       | Reverb                  | \~\$0.20–\$0.40 |
| Deepgram     | Nova-2                  | \~\$0.24        |
| Speechmatics | Any-to-Text Cloud       | \~\$0.24        |

---

## By Approximate Capability (Highest To Lowest)

| Provider     | Model                   | Capability Notes                                         |
| ------------ | ----------------------- | -------------------------------------------------------- |
| OpenAI       | gpt-4o-transcribe       | Integrates LLM reasoning for ASR, strong formatting      |
| Speechmatics | Any-to-Text Cloud       | Excels on accents, robust punctuation and diarization    |
| Deepgram     | Nova-2                  | Very fast, accurate, competitive in benchmarks           |
| Google Cloud | Speech-to-Text v2       | Well-rounded, solid accuracy, good biasing options       |
| AssemblyAI   | Best                    | Strong accuracy + AI extras, stable                      |
| OpenAI       | whisper-1               | High accuracy, robust open-source heritage               |
| AWS          | Transcribe Standard     | Reliable, but slightly behind leaders in accent accuracy |
| Azure        | Speech to Text Standard | Solid, especially in enterprise setups                   |
| Rev AI       | Reverb                  | Decent accuracy for cost, best for budget runs           |
 