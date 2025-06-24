# BrSpeech Training Technical Details

## Architecture

**Model:** W2V+AASIST (Wav2Vec2 SSL features + AASIST graph attention network)  
**Input:** 4-second audio chunks at 16kHz  
**Features:** 768-dim SSL features from Wav2Vec2  
**Classifier:** Graph attention layers + binary classification (bonafide vs spoof)

## Dataset Processing

The pipeline automatically generates metadata CSVs from:
- **Real audio:** BRSpeech corpus (Brazilian Portuguese)
- **Synthetic audio:** 5 TTS models (f5tts, fish-speech, toucantts, xtts, yourtts)
- **Format:** FLAC files, handled natively by torchaudio
- **Labels:** Binary classification (bonafide vs spoof)

## Training Parameters

Based on proven parameters from [are_audio_df_polyglots](https://github.com/bartlomiejmarek/are_audio_df_polyglots):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | `5e-6` | Proven optimal for W2V+AASIST |
| Weight Decay | `5e-7` | Effective regularization |
| Batch Size | `4` | Reduced from original `32` for GPU constraints |
| Epochs | `25` | Sufficient for convergence |
| Optimizer | `Adam` | Standard choice |

## Performance Expectations

- **Dataset sizes:** 444,274 train, 6,956 val, 7,907 test samples
- **Training time:** ~2+ hours per epoch (batch_size=4)
- **Memory usage:** ~7-8GB GPU memory
- **Convergence:** Loss decreases from ~0.9 to ~0.7 in first epoch
- **Accuracy:** 60%+ after first epoch, 90%+ after full training

## Key Differences from Original

1. **Custom Dataset Class:** `BrSpeechDataset` instead of `ASVspoof2019Dataset`
2. **File Format:** Handles FLAC instead of WAV
3. **Metadata Structure:** CSV format instead of ASVspoof2019 protocol files
4. **Custom Training Script:** `train_brspeech.py` adapted for BrSpeech data
5. **Reduced Batch Size:** Optimized for consumer GPU memory

## Troubleshooting

**CUDA Out of Memory:** Reduce `batch_size` in configs/aasist_w2v_brspeech.yaml  
**Dataset not found:** Ensure absolute paths exist and are accessible  
**Import errors:** All dependencies handled within Docker container  
**Permission errors:** Use `sudo` for Docker commands or add user to docker group

## File Structure

```
run_docker.sh                       # Main entry point script
Dockerfile                          # Container definition with training pipeline
docker-compose.yml                  # Container orchestration
src/train_brspeech.py               # Main training logic
src/brspeech_dataset.py             # Custom dataset handling  
src/prepare_brspeech_metadata.py    # Data preprocessing
configs/aasist_w2v_brspeech.yaml    # Model & training config
docs/TRAINING_DETAILS.md            # Technical documentation
```

## Citation

Built upon research from:
```bibtex
@misc{marek2024audiodeepfakedetectionmodels,
      title={Are audio DeepFake detection models polyglots?}, 
      author={Bartłomiej Marek and Piotr Kawa and Piotr Syga},
      year={2024},
      eprint={2412.17924},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2412.17924}, 
}
``` 