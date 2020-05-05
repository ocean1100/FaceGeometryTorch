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
    '--dynamic_landmark_embedding_path',
    type = str,
    default = './model/flame_dynamic_embedding.npy',
    help = 'Dynamic contour embedding path for FLAME'
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

def get_config_with_default_args():
    config = parser.parse_args([])
    return config

def get_config_with_args(cmd_args):
    # cmd_args of the format ['--input', 'PATH_TO_INPUT', '--output', 'PATH_TO_OUTPUT']
    config = parser.parse_args(cmd_args)
    return config