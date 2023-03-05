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


# Training data
DATA_FILE_NAME = "data.csv"
DATA_DIR = os.path.join(PROJECT_PATH, "data")
DATA_FILE = os.path.join(PROJECT_PATH, "data", DATA_FILE_NAME)

# Storing model
MODEL_FILE_NAME = "final_model.sav"
MODEL_DIR = os.path.join(PROJECT_PATH, "ML_modular", "models")
MODEL_FILE = os.path.join(PROJECT_PATH, "ML_modular", "models", MODEL_FILE_NAME)

# Temporary table
TMP_TABLE_FILE_NAME = "tmp_result_table.csv"
TMP_TABLE_DIR = os.path.join(PROJECT_PATH, "ML_modular", "tables")
TMP_TABLE_FILE = os.path.join(PROJECT_PATH, "ML_modular", "tables", TMP_TABLE_FILE_NAME)
