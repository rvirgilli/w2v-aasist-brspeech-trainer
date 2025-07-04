#!/usr/bin/env python
"""
BrSpeech Scores Analysis Script
Analyzes the scores CSV file generated by the test script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import argparse
import seaborn as sns

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER) and optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False Negative Rate
    
    # Find where FPR ≈ FNR
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold, fpr, tpr, thresholds

def analyze_by_source(df):
    """Analyze performance by TTS model source."""
    df['source'] = df['path'].apply(lambda x: extract_source(x))
    
    results = {}
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        if len(source_df) > 0:
            eer, eer_thresh, _, _, _ = calculate_eer(source_df['true_label'], source_df['prediction_score'])
            acc_05 = np.mean((source_df['prediction_score'] > 0.5) == source_df['true_label']) * 100
            results[source] = {
                'count': len(source_df),
                'eer': eer * 100,
                'eer_threshold': eer_thresh,
                'accuracy_05': acc_05
            }
    
    return results

def extract_source(path):
    """Extract TTS model source from file path."""
    if 'f5tts' in path:
        return 'f5tts'
    elif 'fish-speech' in path:
        return 'fish-speech'
    elif 'toucantts' in path:
        return 'toucantts'
    elif 'xtts' in path:
        return 'xtts'
    elif 'yourtts' in path:
        return 'yourtts'
    elif '/real/' in path or 'train/audio' in path:
        return 'real'
    else:
        return 'unknown'

def main():
    parser = argparse.ArgumentParser(description='Analyze BrSpeech test scores')
    parser.add_argument('scores_file', help='Path to scores CSV file')
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory for plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Load scores
    print(f"Loading scores from: {args.scores_file}")
    df = pd.read_csv(args.scores_file)
    
    print(f"Dataset: {len(df)} samples")
    print(f"Real samples: {np.sum(df['true_label'] == 1)}")
    print(f"Fake samples: {np.sum(df['true_label'] == 0)}")
    print()
    
    # Calculate overall EER
    eer, eer_threshold, fpr, tpr, thresholds = calculate_eer(df['true_label'], df['prediction_score'])
    
    # Calculate accuracies
    acc_05 = np.mean(df['predicted_label'] == df['true_label']) * 100
    acc_eer = np.mean((df['prediction_score'] > eer_threshold) == df['true_label']) * 100
    
    # AUC
    auc_score = auc(fpr, tpr)
    
    print("=== OVERALL PERFORMANCE ===")
    print(f"EER: {eer*100:.2f}% (threshold: {eer_threshold:.4f})")
    print(f"AUC: {auc_score:.4f}")
    print(f"Accuracy at threshold 0.5: {acc_05:.2f}%")
    print(f"Accuracy at EER threshold: {acc_eer:.2f}%")
    print()
    
    # Performance by source
    print("=== PERFORMANCE BY SOURCE ===")
    source_results = analyze_by_source(df)
    for source, metrics in source_results.items():
        print(f"{source:15} | Count: {metrics['count']:5} | EER: {metrics['eer']:5.2f}% | Acc@0.5: {metrics['accuracy_05']:5.2f}%")
    print()
    
    # Confusion Matrix at 0.5 threshold
    cm = confusion_matrix(df['true_label'], df['predicted_label'])
    print("=== CONFUSION MATRIX (threshold=0.5) ===")
    print("Predicted:  Real  Fake")
    print(f"Real:       {cm[1,1]:4d}  {cm[1,0]:4d}")
    print(f"Fake:       {cm[0,1]:4d}  {cm[0,0]:4d}")
    print()
    
    # Score statistics
    real_scores = df[df['true_label'] == 1]['prediction_score']
    fake_scores = df[df['true_label'] == 0]['prediction_score']
    
    print("=== SCORE STATISTICS ===")
    print(f"Real speech scores  - Mean: {real_scores.mean():.4f}, Std: {real_scores.std():.4f}")
    print(f"Fake speech scores  - Mean: {fake_scores.mean():.4f}, Std: {fake_scores.std():.4f}")
    print()
    
    # Plotting
    if args.save_plots:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    plt.plot(fpr[np.argmin(np.abs(fpr - (1-tpr)))], tpr[np.argmin(np.abs(fpr - (1-tpr)))], 
             'ro', markersize=8, label=f'EER = {eer*100:.2f}%')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Score distributions
    plt.subplot(1, 2, 2)
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake', color='red', density=True)
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real', color='blue', density=True)
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold 0.5')
    plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER Threshold {eer_threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save_plots:
        plt.savefig(f'{args.output_dir}/analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {args.output_dir}/analysis.png")
    else:
        plt.show()

if __name__ == "__main__":
    main() 