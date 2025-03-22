import torch
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device
