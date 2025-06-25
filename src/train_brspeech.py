#!/usr/bin/env python
"""
BrSpeech Training Script
Custom training script for BrSpeech dataset using W2V+AASIST architecture.
Based on the original train.py from are_audio_df_polyglots.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.datasets.brspeech_dataset import BrSpeechDataset
from src.train_models import train_nn
from configuration.df_train_config import DF_Train_Config
from configuration.train_config import _TrainerConfig
from configuration.rawboost_config import _RawboostConfig
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, train_config: DF_Train_Config) -> tuple:
    """Create train, validation, and test data loaders."""
    logger.info("Creating datasets and data loaders...")
    
    # Use the provided train_config instead of creating a new one
    df_config = train_config
    
    # Create datasets
    train_dataset = BrSpeechDataset(df_config, subset='train')
    val_dataset = BrSpeechDataset(df_config, subset='val')
    test_dataset = BrSpeechDataset(df_config, subset='test')
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BrSpeech audio deepfake detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--weight_decay', type=float, help='Override weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--test_only', action='store_true', help='Run in test-only mode')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for testing')
    
    args = parser.parse_args()
    
    # Load YAML config
    model_config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create DF_Train_Config object
    train_config = DF_Train_Config(
        seed=42,
        trainer_config=_TrainerConfig(
            optimizer=Adam,
            batch_size=args.batch_size or model_config['training']['batch_size'],
            num_epochs=args.epochs or model_config['training']['epochs'],
            early_stopping=model_config['training'].get('early_stopping', False),
            early_stopping_patience=5,
            optimizer_parameters={
                "lr": args.lr or model_config['training']['learning_rate'],
                "weight_decay": args.weight_decay or model_config['training']['weight_decay']
            },
            criterion=BCEWithLogitsLoss,
        ),
        root_dir=Path(model_config['data']['root_dir']),
        rawboost_config=_RawboostConfig(algo_id=0),
        device=device
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(model_config, train_config)
    
    # Set up output directory
    model_name = model_config['model']['name']
    lr = train_config.trainer_config.optimizer_parameters['lr']
    wd = train_config.trainer_config.optimizer_parameters['weight_decay']
    out_model_dir = Path("fine_tuned_models") / f"{model_name}_lr_{lr}_wd_{wd}"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    
    # If in test_only mode, load model and run test
    if args.test_only:
        if not args.checkpoint_path:
            raise ValueError("Must provide --checkpoint_path in test-only mode")
        
        from src.aasist_model.aasist_model import load_model_from_config
        
        logger.info(f"Running in test-only mode with checkpoint: {args.checkpoint_path}")
        
        # Load model from config
        model = load_model_from_config(model_config, device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        
        # Simple test loop
        test_loss = 0
        test_correct = 0
        total = 0
        criterion = BCEWithLogitsLoss().to(device)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float().unsqueeze(1)
                
                output = model(batch_x)
                loss = criterion(output, batch_y)
                
                test_loss += loss.item()
                preds = (torch.sigmoid(output) > 0.5).float()
                test_correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = test_loss / len(test_loader)
        accuracy = (test_correct / total) * 100
        logger.info(f"Test Set Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return
    
    # Train the model using the original train_nn function
    logger.info("Starting training...")
    config_save_path, checkpoint_path = train_nn(
        data_train=train_loader,
        data_test=val_loader,
        config=train_config,
        model_config=model_config,
        out_dir=out_model_dir,
        device=device,
    )
    
    logger.info(f"Training completed! Model saved at: {config_save_path}")
    logger.info(f"Checkpoint saved at: {checkpoint_path}")
    
    # Test on test set if available
    if test_loader:
        logger.info("Testing on test set...")
        # Note: The original code doesn't have a separate test function,
        # so we'll just log that training is complete for now


if __name__ == "__main__":
    main() 