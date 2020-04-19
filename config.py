import argparse
import os

parser = argparse.ArgumentParser(description = 'FLAME model')

parser.add_argument(
    '--flame_model_path',
    type = str,
    default = './model/generic_model.pkl',
    help = 'flame model path'
)

parser.add_argument(
    '--static_landmark_embedding_path',
    type = str,
    default = './data/flame_static_embedding.pkl',
    help = 'Static landmark embeddings path for FLAME'
)
parser.add_argument(
    '--batch_size',
    type = int,
    default = 8,
    help = 'Training batch size.'
)

def get_config():
    config = parser.parse_args()
    return config
