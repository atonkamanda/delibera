#!/usr/bin/env python
"""
main.py – Single GPU Pipeline for Deliberative Alignment

This entry point supports three stages:
  1. Filtering (data generation via sft_data_generation)
  2. Supervised Fine Tuning (SFT via sft)
  3. Reinforcement Learning (RL via rl)

Each stage is run on a single GPU.
"""

import argparse
import os
import yaml

# Import the modules for each stage.
from src import sft_data_generation
from src import sft
from src import rl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--stage', type=str, choices=["filter", "sft", "rl", "pipeline"],
                        default="pipeline", help="Which stage to run: filter, sft, rl, or pipeline")
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='List of devices to use (default single GPU).')
    args = parser.parse_args()
    
    # Load full configuration from the YAML file.
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # For the RL stage, use the SFT checkpoint as the starting point.
    full_config['rl_model_path'] = "sft_model"
    
    # For a single GPU scenario, use the first device.
    device = args.devices[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(":")[-1]
    
    if args.stage == "filter":
        sft_data_generation.run_filtering_stage(full_config)
    elif args.stage == "sft":
        sft.run_sft_stage(full_config)
    elif args.stage == "rl":
        rl.train_rl(full_config)
    elif args.stage == "pipeline":
        print("Starting Data Generation Stage")
        sft_data_generation.run_filtering_stage(full_config)
        print("Starting SFT Stage")
        sft.run_sft_stage(full_config)
        print("Starting RL Stage")
        rl.train_rl(full_config)

if __name__ == '__main__':
    main()