from pathlib import Path

SWEEP_SETTINGS_JSON = Path("./data/sweep_settings_dict.json") # map to saved paramters from the hyper-param search
CRISPR_MODE = "concat"
RANDOM_STATE = 41
PROJ_VERSION = "AcrTransAct_v5"
DEBUG_MODE = False
BS = 32
EPOCHS= 250
TEST_SIZE = 0.2  
UNDERSAMPLE = False
MONITOR_LR = "val_loss"
REDUCE_LR_FACTOR = 0.9
REDUCE_LR_TECHNIQUE = "plateau"
OPTIMIZE_METRIC = "F1_CV"

# CROSS VAL
EPOCHS_CV = 100
REPEAT_CV = 3  
CV_FOLDS = 5

SWEEP_RUNS = 20 # number of runs to do hyperparameter tuning

# DATA
VERSION_FOLDER = Path(f"./data/")
INHIBITION_EXCEL_FILE = Path("Acr_CRISPR_inhibition_type_I.xlsx")
INHIBITION_DF_PATH = Path(VERSION_FOLDER, INHIBITION_EXCEL_FILE)