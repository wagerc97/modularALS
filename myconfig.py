#!/usr/bin/python3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
myconfig.py holds project config 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os

configpath = os.path.abspath(__file__).split("myconfig.py")[0]

def testiamhere():
    print("\n")
    print("+" * 30)
    print(f"{__name__} is here")
    print("Project path=" + configpath)
    print("+" * 30, "\n")


# Training data
TRAIN_DATA_DIR = os.path.join(configpath, "data")
TRAIN_DATA_FILE_NAME = "data.csv"
TRAIN_DATA_FILE = os.path.join(configpath, "data", TRAIN_DATA_FILE_NAME)

# Storing model
MODEL_DIR = os.path.join(configpath, "ML_modular", "models")
MODEL_FILE_NAME = "final_model.sav"
MODEL_FILE = os.path.join(configpath, "ML_modular", "models", MODEL_FILE_NAME)

# Temporary table
TMP_TABLE_DIR = os.path.join(configpath, "ML_modular", "tables")
TMP_TABLE_FILE_NAME = "tmp_result_table.csv"
TMP_TABLE_FILE = os.path.join(configpath, "ML_modular", "tables", TMP_TABLE_FILE_NAME)
