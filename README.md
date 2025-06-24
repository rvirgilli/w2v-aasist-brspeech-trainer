# BrSpeech DeepFake Detection Training

> Docker-based training pipeline for audio deepfake detection using Brazilian Portuguese data with W2V+AASIST models.

## Overview

This repository provides a complete Docker-based training pipeline for training W2V+AASIST models on the BrSpeech dataset for audio deepfake detection. It builds upon the [are_audio_df_polyglots](https://github.com/bartlomiejmarek/are_audio_df_polyglots) codebase with custom adaptations for Brazilian Portuguese audio data.

## Features

- ✅ **Docker-based setup** with NVIDIA GPU support
- ✅ **Custom BrSpeech dataset handling** with automatic metadata generation
- ✅ **W2V+AASIST model architecture** with pretrained Wav2Vec2 features
- ✅ **YAML-based configuration** for easy parameter tuning
- ✅ **Tested and working** on real BrSpeech datasets (444K+ samples)

## Quick Start

### Prerequisites
- Docker with NVIDIA Container Runtime
- NVIDIA GPU with 11GB+ memory
- BrSpeech datasets (real + synthetic)

### Run Training

```bash
# Make script executable
chmod +x run_docker.sh

# Start training with your dataset paths
sudo ./run_docker.sh /path/to/BRSpeech_CML_TTS_v04012024 /path/to/brspeech_df ./outputs --build
```

### Expected Dataset Structure
```
BRSpeech_CML_TTS_v04012024/     # Real speech data
├── train.csv                   # Pipe-delimited metadata
├── dev.csv
├── test.csv
└── train/audio/                # Audio files (.flac)

brspeech_df/                    # Synthetic speech data
├── f5tts/
├── fish-speech/
├── toucantts/
├── xtts/
└── yourtts/
    └── {train,dev,test}/audio/ # Audio files (.flac)
```

## Configuration

Edit `configs/aasist_w2v_brspeech.yaml` to modify training parameters:

```yaml
training:
  batch_size: 4           # Adjust based on GPU memory
  learning_rate: 5e-6     # Proven optimal value
  weight_decay: 5e-7      # Regularization
  epochs: 25              # Training duration
  optimizer: Adam
  early_stopping: false
```

## Key Files

| File | Purpose |
|------|---------|
| `run_docker.sh` | **Main entry point** - handles dataset mounting and Docker execution |
| `src/train_brspeech.py` | **Custom training script** - adapted for BrSpeech dataset |
| `configs/aasist_w2v_brspeech.yaml` | **Model configuration** - training parameters and architecture |
| `src/prepare_brspeech_metadata.py` | **Dataset processing** - generates training metadata CSVs |
| `src/brspeech_dataset.py` | **Custom dataset class** - handles BrSpeech data loading |

## Results

- **Dataset sizes:** 444,274 train, 6,956 val, 7,907 test samples
- **Training time:** ~2+ hours per epoch (batch_size=4)
- **Expected accuracy:** 60%+ after first epoch, 90%+ after full training
- **Outputs:** Model checkpoints saved to `./outputs/w2v_aasist_lr_5e-06_wd_5e-07/`

## Technical Details

**Model:** W2V+AASIST (Wav2Vec2 SSL features + AASIST graph attention network)  
**Input:** 4-second audio chunks at 16kHz  
**Architecture:** Based on proven parameters from multilingual audio deepfake research  
**Training:** 25 epochs with Adam optimizer and BCEWithLogitsLoss

## Troubleshooting

**CUDA Out of Memory:** Reduce batch_size in `configs/aasist_w2v_brspeech.yaml`  
**Docker permission errors:** Use `sudo` or add user to docker group  
**Dataset not found:** Check absolute paths exist before running

## References

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

## License

This project builds upon [are_audio_df_polyglots](https://github.com/bartlomiejmarek/are_audio_df_polyglots). Please refer to the original repository for licensing information.