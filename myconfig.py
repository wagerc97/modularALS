#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
myconfig.py holds project config 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os

PROJECT_PATH = os.path.abspath(__file__).split("myconfig.py")[0]

def testiamhere():
    print("\n")
    print("+" * 30)
    print(f"{__name__} is here")
    print("Project path=" + PROJECT_PATH)
    print("+" * 30, "\n")


### The data directory ###
DATA_DIR = os.path.join(PROJECT_PATH, "data")
# Training data (to train model with)
TRAIN_DATA_NAME = "train.csv"
TRAIN_DATA_FILE = os.path.join(PROJECT_PATH, "data", TRAIN_DATA_NAME)
# Test data (to predict on)
PRED_DATA_NAME = "pred.csv"
PRED_DATA_FILE = os.path.join(PROJECT_PATH, "data", PRED_DATA_NAME)

# Storing model
MODEL_FILE_NAME = "final_model.sav"
MODEL_DIR = os.path.join(PROJECT_PATH, "ML", "models")
MODEL_FILE = os.path.join(PROJECT_PATH, "ML", "models", MODEL_FILE_NAME)

# Temporary table
TMP_TABLE_FILE_NAME = "tmp_result_table.csv"
TMP_TABLE_DIR = os.path.join(PROJECT_PATH, "ML", "tables")
TMP_TABLE_FILE = os.path.join(PROJECT_PATH, "ML", "tables", TMP_TABLE_FILE_NAME)

# Datagenerator modes
_modeTRAIN = "train"
_modePREDICT = "predict"
