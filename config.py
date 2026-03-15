import os


HR_SIZE = 128
LR_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4

# --- PATHS ---
DATASET_DIR = "data"
WEIGHTS_DIR = "saved_models"
LOGS_DIR = "logs"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)