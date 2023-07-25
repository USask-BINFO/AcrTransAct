from pathlib import Path

SWEEP_SETTINGS_JSON = "code/results/AcrTransAct_v5/sweep_settings_dict.json"
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

# CROSS VAL
EPOCHS_CV = 100
REPEAT_CV = 3  
CV_FOLDS = 5
SWEEP_RUNS = 20 # number of runs to do hyperparameter tuning

# DATA
DATA_VERSION = "v1.4.15.2"
VERSION_FOLDER = Path(f"data/Interaction_datasets/versions/{DATA_VERSION}")
INHIBITION_EXCEL_FILE = (
    "Acr_CRISPR_inhibition_type_I_noIC1.xlsx"
    if PROJ_VERSION == "AcrTransAct_v3.2"
    else "Acr_CRISPR_inhibition_type_I.xlsx"
)
INHIBITION_DF_PATH = Path(VERSION_FOLDER, INHIBITION_EXCEL_FILE)
