#!/usr/bin/env python
"""
FSOI Evaluation Script

Evaluate FSOI on a trained model checkpoint without retraining.
Uses sequential sampling on validation set to enable FSOI with
sequential background (x_b = forecast from previous batch).

Usage:
    python evaluate_fsoi.py --checkpoint path/to/checkpoint.ckpt

Example:
    python evaluate_fsoi.py \
        --checkpoint /scratch3/NCEPDEV/da/Azadeh.Gholoubi/add_fsoi/ocelot/gnn_model/test1/last.ckpt \
        --fsoi_batches 5 \
        --output_dir fsoi_evaluation_test1
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate FSOI on trained checkpoint')
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', 
        required=True,
        help='Path to checkpoint file (e.g., /path/to/last.ckpt)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--fsoi_batches',
        type=int,
        default=5,
        help='Number of validation batches to compute FSOI on (default: 5)'
    )
    parser.add_argument(
        '--output_dir',
        default='fsoi_evaluation',
        help='Directory to save FSOI results (default: fsoi_evaluation)'
    )
    parser.add_argument(
        '--conventional_only',
        action='store_true',
        help='Only compute FSOI for conventional obs (radiosonde)'
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=None,
        help='Limit total validation batches (default: same as fsoi_batches)'
    )
    
    args = parser.parse_args()
    
    # Set limit_val_batches to fsoi_batches if not specified
    if args.limit_val_batches is None:
        args.limit_val_batches = args.fsoi_batches
    
    print("=" * 60)
    print("FSOI EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Output dir:        {args.output_dir}")
    print(f"FSOI batches:      {args.fsoi_batches}")
    print(f"Conventional only: {args.conventional_only}")
    print(f"Val batches:       {args.limit_val_batches}")
    print("=" * 60)
    print()
    
    # Build command-line arguments for train_gnn.py
    train_args = [
        "train_gnn.py",
        "--resume_from_checkpoint", args.checkpoint,
        "--sampling_mode", "sequential",
        "--window_mode", "sequential",
        "--enable_fsoi",
        "--fsoi_batches", str(args.fsoi_batches),
        "--fsoi_every_n_epochs", "1",
        "--fsoi_start_epoch", "0",
        "--default_root_dir", args.output_dir,
        "--max_epochs", "1",
        "--limit_train_batches", "0",  # Skip training
        "--limit_val_batches", str(args.limit_val_batches),
        "--devices", "1",  # Force single device to avoid DDP issues
    ]
    
    if args.conventional_only:
        train_args.append("--fsoi_conventional_only")
    
    # Replace sys.argv and call train_gnn.py
    sys.argv = train_args
    
    print("Running FSOI evaluation...")
    print(f"Command: python {' '.join(train_args)}")
    print()
    
    # Import and run train_gnn
    from train_gnn import main as train_main
    train_main()
    
    print()
    print("=" * 60)
    print("FSOI EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/fsoi_results_conventional/")
    print()
    print("To view results:")
    print(f"  cd {args.output_dir}/fsoi_results_conventional/detailed")
    print("  head -20 fsoi_epoch0_batch0.csv  # First batch (climatological)")
    print("  head -20 fsoi_epoch0_batch1.csv  # Second batch (sequential)")
    print()
    print("To analyze:")
    print(f"  python analyze_fsoi_results.py {args.output_dir}/fsoi_results_conventional")
    print()


if __name__ == "__main__":
    main()
