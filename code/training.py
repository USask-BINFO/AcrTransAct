#!/usr/bin/env python
# coding: utf-8
import wandb
import random
import os
import pickle
import json
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.data_utils import *
from utils.model_utils import *
from utils.ppi_utils import *
from utils.logger import Logger
from utils.models.cnn_model import AcrTransAct_CNN
from utils.models.lstm_model import AcrTransAct_LSTM
from sklearn.utils import class_weight
from datetime import datetime
from utils.common_vars import *
import shutil
from argparse import ArgumentParser
import time
from utils.misc import *

seed_everything(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ Arguments ################
parser = ArgumentParser()
# parser.add_argument("--sweep_id", type=str, default="rfkjw8l3")
parser.add_argument("--model_type", type=str, default="CNN", help="CNN or LSTM")
parser.add_argument("--esm_version", type=str, default="35m")
parser.add_argument("--feature_mode", type=int, default=1,
    help="1: ESM, 2: SF, 3: one-hot AA, 4: SF + ESM",
)
parser.add_argument("--do_sweep", type=bool, default=False)
parser.add_argument("--wandb_log", type=bool, default=False)
parser.add_argument("--cross_val", type=bool, default=False)
parser.add_argument("--undersample", type=bool, default=False)
parser.add_argument("--optimize_metric", type=str, default="F1_CV")
parser.add_argument("--monitor_ckpt", type=str, default="val_f1_score")
parser.add_argument("--excl_mode", type=int, default=0,
                    help="0: no exclusion, 1: exclude K12, ATCC_IF, 2: exclude K12, PaLML1_DVT419",
)
parser.add_argument("--run_mode", type=str, default="eval", help="train or eval")

args = parser.parse_args()
# sweep_id = args.sweep_id
model_type = args.model_type
feature_mode = args.feature_mode
esm_version = args.esm_version
do_sweep = False#args.do_sweep
wandb_log = args.wandb_log
cross_val = args.cross_val
undersample = args.undersample
excl_mode = args.excl_mode
optimize_metric = args.optimize_metric
monitor_ckpt = args.monitor_ckpt
run_mode = args.run_mode
print(args)

if wandb_log:
    os.environ["WANDB_SILENT"] = "true"

if run_mode =="eval":
    do_sweep = False
    wandb_log = False
    cross_val = False
    
time_stamp = datetime.now().strftime("%m-%d_%H:%M:%S")
logger = Logger(f"./code/logs/", f"logs_{time_stamp}.log")
logger.log(f"time stamp: {time_stamp}")
logger.log(f"debug mode: {DEBUG_MODE}")
logger.log(f"args: {args}")

exclude_mode_dict = {
    0: None,
    1: ["K12", "ATCC39006_IF"],
    2: ["K12", "PaLML1_DVT419"],
}
if monitor_ckpt.lower().__contains__("f1"):
    opt = "f1"
elif monitor_ckpt.lower().__contains__("loss"):
    opt = "loss"

features_aa, features_sf = None, None
if feature_mode == 1:
    use_sf = False
    FE_name = f"ESM{esm_version}"
    use_aa = True
elif feature_mode == 2:
    use_sf = True
    FE_name = "Just SF"
    use_aa = False
elif feature_mode == 3:
    use_sf = False
    FE_name = "One-hot AA"
    use_aa = True
elif feature_mode == 4: # ESM + SF
    use_sf = True
    use_aa = True
    FE_name = f"ESM{esm_version}"

######################################## Read Data ########################################
logger.log(f"reading data from: {INHIBITION_EXCEL_FILE}")
model_type_dic = {"CNN": AcrTransAct_CNN, "LSTM": AcrTransAct_LSTM}
features_config = return_features_config(FE_name, task="PPI")

if feature_mode == 4:
    features_names = FE_name + " + SF"
else:
    features_names = FE_name

logger.log(f"data version: {DATA_VERSION}")
logger.log(f"use_sf: {use_sf}")
logger.log(f"use_aa: {use_aa}")
logger.log(f"features names: {FE_name}")
logger.log(f"project name: {PROJ_VERSION}")

inhibition_df = pd.read_excel(INHIBITION_DF_PATH)
inhibition_df = inhibition_df[inhibition_df["use"] == 1]
logger.log(f"Number of samples that can be used: {len(inhibition_df)}")

if undersample:
    inhibition_df = undersample_PaLML1_SMC4386(inhibition_df)
    logger.log(f"undersampling PaLML1 and SMC4386, new data size:{len(inhibition_df)}")

if exclude_mode_dict[excl_mode] is not None:
    inhibition_df = inhibition_df[
        ~inhibition_df["CRISPR_name_short"].isin(exclude_mode_dict[excl_mode])
    ]

CRISPR_df = pd.read_excel(VERSION_FOLDER / "CRISPR" / "CRISPR_df.xlsx")
data = read_ppi_data_df(
    inhibition_df,
    CRISPR_df,
    verbose=0,
)

feature_dir = f"{VERSION_FOLDER}/pkl/{PROJ_VERSION}/"
features_file_name = f"_{FE_name}_rs{RANDOM_STATE}.pkl"

if exclude_mode_dict[excl_mode] is not None:
    features_file_name = features_file_name.replace(
        ".pkl", f"_excl_{'_'.join(exclude_mode_dict[excl_mode])}.pkl"
    )
if undersample:
    features_file_name = features_file_name.replace(".pkl", "_undersample.pkl")

###################### Load input features ######################
# AA features
if feature_mode == 3:
    features_config[
        "feature_mode"
    ] = 1  # 0 for just ESM | 1 for one-hot encoding of AA| 3 for just sf |
    features_aa = extract_combined_features(data, features_config, verbose=1)
    logger.log("Using the raw AA")

# Structural Features
if feature_mode==2 or feature_mode==4:
    features_config["feature_mode"] = 3
    os.makedirs(feature_dir, exist_ok=True)
    ffn_sf = "SF" + features_file_name
    if os.path.exists(feature_dir + ffn_sf) != True:
        features_sf = extract_combined_features(data, features_config, verbose=1)
        with open(feature_dir + ffn_sf, "wb") as f:
            pickle.dump(features_sf, f)
    else:
        features_sf = pickle.load(open(feature_dir + ffn_sf, "rb"))
    logger.log("Using SF")

# ESM features 
if feature_mode==1 or feature_mode==4:
    features_config[
        "feature_mode"
    ] = 0 
    os.makedirs(feature_dir, exist_ok=True)
    ffn_trans = (
        "ESM" + features_file_name
        if features_config["model_name"].find("ESM") != -1
        else "prot" + features_file_name
    )

    if os.path.exists(feature_dir + ffn_trans) != True:
        load_feature_extractor(features_config, return_model=True)
        features_aa = extract_combined_features(data, features_config, verbose=1)
        with open(feature_dir + ffn_trans, "wb") as f:
            pickle.dump(features_aa, f)
    else:
        features_aa = pickle.load(open(feature_dir + ffn_trans, "rb"))

    logger.log("Using ESM features")

############################## DATA SPLIT ##############################
split_dict = split_data(
    inhibition_df, data, features_aa, features_sf, test_size=TEST_SIZE
)

train_df = split_dict["train_df"]
test_df = split_dict["test_df"]
train_data = split_dict["train_data"]
test_data = split_dict["test_data"]

if use_aa:
    X_train_aa = split_dict["X_train_aa"]
    X_test_aa = split_dict["X_test_aa"]
if use_sf:
    X_train_sf = split_dict["X_train_sf"]
    X_test_sf = split_dict["X_test_sf"]

logger.log(f"Number of samples in train: {len(train_data)}")
logger.log(f"Number of samples in test: {len(test_data)}")

y_train = [d["inhibition"] for d in train_data]
y_test = [d["inhibition"] for d in test_data]

loader_input = {
    "X_train_aa": X_train_aa if use_aa else None,
    "X_val_aa": X_test_aa if use_aa else None,
    "X_train_sf": X_train_sf if use_sf else None,
    "X_val_sf": X_test_sf if use_sf else None,
    "y_train": y_train,
    "y_val": y_test,
}

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = torch.tensor(
    class_weights, device=features_config["device"], dtype=torch.float
)

plot_train_val_data(
    train_data,
    test_data,
    save_path=f"{VERSION_FOLDER}/label_dist.jpg",
    plot=0,
)

loaders = return_loaders(
    loader_input,
    use_aa=use_aa,
    use_sf=use_sf,
    bs=BS,
    channels_first=features_config["channels_first"],
    val=True,
    device=device
)

i = 1 if features_config["channels_first"] else 0
if use_sf and use_aa:
    seq_len_aa = loaders["train_loader"].dataset[0][0][0].size()[i]
    seq_len_sf = loaders["train_loader"].dataset[0][0][1].size()[i]
elif use_aa and not use_sf:
    seq_len_aa = loaders["train_loader"].dataset[0][0].size()[i]
    seq_len_sf = 0
elif use_sf and not use_aa:
    seq_len_aa = 0
    seq_len_sf = loaders["train_loader"].dataset[0][0].size()[i]
######################################## CROSS VALIDATION FUNCTION ########################################

def cross_validation(cls_config, cls_name, reps=1, folds=5, epochs=100, tune=False):
    """
    :param tune: whether this function is being used whithin the wandb sweep search
    """
    global ckpt_dir_cv

    kf_stratified = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    labels_CRISPR_sys = [
        str(d["inhibition"]) + "_" + d["CRISPR_system"] for d in train_data
    ]
    cv_rep_res = []
    labels = [d["inhibition"] for d in train_data]

    if not tune:
        logger.log("Cross validation started...")
    for rep in range(reps):
        cv_folds_res = []
        for fold, (train_index, test_index) in enumerate(
            kf_stratified.split(
                X=X_train_aa if use_aa else X_train_sf, y=labels_CRISPR_sys
            )
        ):
            if not tune and not DEBUG_MODE:
                print(f"> ****** Fold {fold+1} of {folds} | Rep: {rep+1}/{reps} ******")
            fold_name = f"fold_{fold+1}_rep{rep+1}"
            ######################
            ## DATA PREPARATION ##
            ######################
            if use_aa:
                X_train_aa_fold, X_val_aa_fold = [X_train_aa[i] for i in train_index], [
                    X_train_aa[i] for i in test_index
                ]
            if use_sf:
                X_train_sf_fold, X_val_sf_fold = [X_train_sf[i] for i in train_index], [
                    X_train_sf[i] for i in test_index
                ]
            y_train, y_val = [labels[i] for i in train_index], [
                labels[i] for i in test_index
            ]
            data_val = [data[i] for i in test_index]

            train_val_data = {
                "X_train_aa": X_train_aa_fold if use_aa else None,
                "X_val_aa": X_val_aa_fold if use_aa else None,
                "X_train_sf": X_train_sf_fold if use_sf else None,
                "X_val_sf": X_val_sf_fold if use_sf else None,
                "y_train": y_train,
                "y_val": y_val,
            }

            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(
                class_weights, device=features_config["device"], dtype=torch.float
            )
            cls_config["class_weights"] = class_weights
            loaders = return_loaders(
                train_val_data, bs=BS, use_sf=use_sf, use_aa=use_aa
            )

            aa_features_shape, ss_features_shape = return_aa_ss_shapes(
                loaders, use_aa, use_sf
            )
            ###################
            ## MODEL CONFIGS ##
            ###################
            i = 1 if features_config["channels_first"] else 0
            cls_config["seq_len_aa"] = aa_features_shape[i] if use_aa else 0
            cls_config["seq_len_sf"] = ss_features_shape[i] if use_sf else 0
            cls_config["hidden_size_aa"] = aa_features_shape[i - 1] if use_aa else 0
            cls_config["hidden_size_sf"] = ss_features_shape[i - 1] if use_sf else 0
            if tune:
                ckpt_dir_cv = f"./code/results/{PROJ_VERSION}/temp_chkpts_tune/"
            
            chkpt_dir = f"{ckpt_dir_cv}/{cls_name}_{fold_name}_{time_stamp}/"


            if exclude_mode_dict[excl_mode] is not None:
                chkpt_dir = chkpt_dir[:-1]+f"_excl{'_'.join(exclude_mode_dict[excl_mode])}"
            else:
                chkpt_dir = chkpt_dir[:-1]+f"_no_excl"

            os.makedirs(chkpt_dir, exist_ok=True)
            chkpt_name = "best_model"

            if os.path.isfile(chkpt_dir + chkpt_name):
                chkpt_dir = chkpt_dir[:-1] + str(random.randint(0, 1000)) + "/"

            chkpt_call = ModelCheckpoint(
                dirpath=chkpt_dir,
                filename=chkpt_name,
                save_top_k=1,
                monitor=monitor_ckpt,
                mode="min" if monitor_ckpt.__contains__("loss") else "max",
                save_weights_only=True,
            )
            #############################
            ## MODEL TRAINING AND EVAL ##
            #############################
            trainer = pl.Trainer(
                max_epochs=epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                accelerator="gpu",
                devices=1,
                callbacks=[
                    chkpt_call,
                ],
                max_steps=1 if DEBUG_MODE else -1,
            )
            model = model_type_dic[model_type](
                cls_config,
                channels_first=features_config["channels_first"],
            )
            trainer.fit(
                model,
                loaders["train_loader"],
                loaders["val_loader"],
            )
            test_results = trainer.test(
                model,
                loaders["val_loader"],
                ckpt_path=chkpt_call.best_model_path if not DEBUG_MODE else None,
                verbose=0,
            )[0]
            preds = model.pred_val(
                loaders["val_loader"], use_aa, use_sf, return_probs=True
            )
            preds_pos = preds[:, 1]
            auc_dict = return_AUC(
                preds_pos,
                y_val,
            )
            pred_results = {
                "f1": np.round(test_results["test_f1"], 2),
                "precision": np.round(test_results["test_precision"], 2),
                "recall": np.round(test_results["test_recall"], 2),
                "accuracy": np.round(test_results["test_acc"], 2),
                "loss": np.round(test_results["test_loss"], 2),
                "auc": np.round(auc_dict["AUC"], 2),
                "aupr": np.round(auc_dict["AUPR"], 2),
            }
            preds_stats = prediction_stats(preds, data_val)

            if not tune:
                print(f"> RESULTS: {pred_results}, BEST EPOCH: {model.best_epoch}")
                logger.log(
                    f"fold {fold+1}/{CV_FOLDS} rep {rep+1}/{REPEAT_CV} results:\n{pred_results}, best epoch: {model.best_epoch}"
                )
            ######################
            ## LOGGING RESULTS ##
            ######################
            # fold results for when retriving each fold
            cv_fold_res = {
                "Fold": fold + 1,
                "Rep": rep + 1,
                "seq_len_aa": cls_config["seq_len_aa"],
                "seq_len_sf": cls_config["seq_len_sf"],
                "check_point": f"{chkpt_dir}/{chkpt_name}.ckpt",
                "sweep_id": sweep_id,
                "preds_stats": preds_stats,
            }
            # adding the info in pred_results to cv_fold_res
            for key in pred_results.keys():
                cv_fold_res[key] = pred_results[key]

            cv_folds_res.append(cv_fold_res)  # fold results

            del model
            del loaders
            torch.cuda.empty_cache()

        cv_rep_res.append(cv_folds_res)  # rep results

    return cv_rep_res

######################################## HYPERPARAMETER TUNING ########################################
if do_sweep:
    sweep_i = 1
    sweep_config = {
        "name": f"{model_type}_{time_stamp}",
        "method": "bayes", 
        "metric": {
            "name": optimize_metric,
            "goal": "maximize"
            if optimize_metric.lower().__contains__("f1")
            else "minimize",
        },
        "parameters": {
            "lr": {"min": 3e-4, "max": 3e-3},
            "weight_decay": {"min": 1e-3, "max": 1e-2},
            "dout": {"min": 0.3, "max": 0.5},
            "kernel_size_aa": {"values": [7] if use_aa else [0]},
            "kernel_size_sf": {"values": [7] if use_sf else [0]},
            "out_channels_sf": {"values": [4, 8] if use_sf else [0]},
            "out_channels_aa": {"values": [4, 8] if use_aa else [0]},
            "batch_size": {"values": [BS]},
            "FC_nodes": {"values": [4, 8]},
            "mode": {"values": [2]},
            "optimizer": {"values": ["Adam"]},
            "use_sf": {"values": [use_sf]},
            "use_aa": {"values": [use_aa]},
            "num_main_layers": {"values": [1, 2]},
        },
    }

    logger.log(f"starting the search, optimizing for: {sweep_config['metric']}")

    def train_hparam_search():
        global sweep_i
        wandb.init(project=PROJ_VERSION)
        logger.log(f"starting sweep {sweep_i}")
        sweep_i += 1
        cls_config_tuning = get_config_from_pre_tune(
            wandb.config, monitor_ckpt, model_type
        )
        wb_logger = WandbLogger(log_model=False)
        
        cls_model_name = return_PPI_name(cls_config_tuning, model_type)
        
        hparams = {
            "Model Name": cls_model_name,
            "Feature Extractor": features_config["model_name"],
            "Data Version": DATA_VERSION,
        }
        wb_logger.log_hyperparams(hparams)
        
        cv_rep_res = cross_validation(
            cls_config_tuning,
            reps=1,
            folds=CV_FOLDS,
            epochs=50,
            tune=True,
            cls_name=cls_model_name,
        )
        avg_results_cv = return_avg_cv_results(cv_rep_res)
        
        wb_logger.log_metrics(avg_results_cv)
        wandb.finish()

    #RUN SWEEP
    free_mem()
    sweep_id = wandb.sweep(sweep_config, project=PROJ_VERSION)
    start = time.time()
    wandb.agent(sweep_id, train_hparam_search, count=SWEEP_RUNS if not DEBUG_MODE else 1)
    end = time.time()
    logger.log(f"total search time: {(end-start)/60:.2f} minutes")

################################################## TRAINING ##################################################
# load the model's config
sweep_key = model_type + "_" + features_names
with open(SWEEP_SETTINGS_JSON, "r") as f:
    sweep_settings = json.load(f)
    sweep_id = sweep_settings[sweep_key]
logger.log(f"using configs from sweep_id: {sweep_id}, read from disk")

# TODO change to PROJ_VERSION
inference_dir =  f"./code/results/AcrTransAct_v5/inf_{opt}_opt_{model_type}"+\
                 f"/{sweep_id if sweep_id is not None else 'no_sweep'}/"
if exclude_mode_dict[excl_mode] is not None:
    inference_dir = inference_dir[:-1]+f"_excl_{'_'.join(exclude_mode_dict[excl_mode])}/"
else:
    inference_dir = inference_dir[:-1]+"_no_excl/"
os.makedirs(inference_dir, exist_ok=True)

with open(f"{inference_dir}/config.json", "r") as f:
    cls_config = json.load(f)

cls_config["class_weights"] = class_weights 
cls_model_name = cls_config["cls_model_name"]

save_cv_dir = f"./code/results/{PROJ_VERSION}/cv_rep_res/{sweep_id}/"
os.makedirs(save_cv_dir, exist_ok=True)
logger.log(f"Cross val save dir: {save_cv_dir}")
    
ckpt_dir_cv = (
    f"./code/results/{PROJ_VERSION}/temp_chkpts/"
)

logger.log(f"inference_dir: {inference_dir}")
logger.log(f"ckpt_dir_cv: {ckpt_dir_cv}")

#################### CROSS VALIDATION ####################
if wandb_log:
    logger.log("WandB Logging started ...")
    wb_logger = WandbLogger(
        project=PROJ_VERSION,
        name=cls_model_name,
        log_model=False,
    )
if cross_val:
    cv_rep_res = cross_validation(
        cls_config, cls_model_name, reps=REPEAT_CV, folds=CV_FOLDS
    )
    avg_results_cv = return_avg_cv_results(cv_rep_res)

    # average accuracy for each crispr system for all reps
    reps_crispr_acc = each_CRISPR_results(cv_rep_res)
    avg_acc = avg_each_CRISPR_results(reps_crispr_acc)
    print("Average Accuracy for Each System:")
    logger.log("Average Accuracy for Each System:")
    for system, avg_accuracy in avg_acc.items():
        print(system + ": " + str(avg_accuracy))
        logger.log(system + ": " + str(avg_accuracy))

    if wandb_log:
        wandb.log(
            {"cv_folds": CV_FOLDS,
            "Kfold_repeats": REPEAT_CV,
            }
        )
        wandb.log(avg_results_cv)
        with open(f"{save_cv_dir}/cv_config.json", "w") as f:
            json.dump(cv_rep_res, f)
#################### TRAIN AND TEST, NO CROSSVAL ####################
logger.log(f"model name: {cls_model_name}...")

if run_mode == "train":
    ckpt_name = "best_model"+"_"+time_stamp
else:
    ckpt_name = cls_config["weights"].replace(".ckpt", "")

chkpt_call = ModelCheckpoint(
    inference_dir,
    filename=ckpt_name,
    monitor=monitor_ckpt,
    save_top_k=1,
    mode="max" if monitor_ckpt.lower().__contains__("f1") else "min",
    verbose=0,
)

es_call = EarlyStopping(
    monitor=monitor_ckpt if not DEBUG_MODE else "train_loss",
    patience=30,
    mode="min" if monitor_ckpt.__contains__("loss") else "max",
    verbose=1,
    min_delta=0.005,
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_progress_bar=True,
    enable_model_summary=True,
    accelerator="gpu",
    devices=1,
    max_steps=1 if DEBUG_MODE else -1,
    logger=wb_logger if wandb_log else None,
    callbacks=[chkpt_call, es_call],
)
model = model_type_dic[model_type](
        cls_config,
        channels_first=features_config["channels_first"],
        verbose=0,
        debug_mode=0,
        )
# train on train-val set
if run_mode == "train":
    logger.log("#" * 50)
    logger.log(f"Training started on {len(train_data)} samples...")
    logger.log(f"validation size: {len(test_data)}")
    logger.log(f"checkpoint metric: {monitor_ckpt}")

    trainer.fit(
        model,
        loaders["train_loader"],
        loaders["val_loader"],
    )
    logger.log(f"checkpoint at: {chkpt_call.best_model_path}")

    print("Training finished ...")
    os.makedirs(inference_dir, exist_ok=True)
    model.plot_history(
    running_mean_window=1,
    save_path=f"{inference_dir}/{time_stamp}_history.png",
    plot=False,
)

model = model_type_dic[model_type].load_from_checkpoint(
    inference_dir + f"/{ckpt_name}.ckpt",
    cls_config=cls_config,
    channels_first=features_config["channels_first"],
    verbose=0,
    debug_mode=0,
)
model.eval()

preds = model.pred_val(
    loaders["val_loader"],
    return_probs=True,
    use_aa=use_aa,
    use_sf=use_sf,
)
preds_pos = preds[:, 1]
auc_dict = return_AUC(
    preds_pos,
    y_test,
    save_path=inference_dir,
    plot_ROC=False,
)
pred_stats = prediction_stats(preds, test_data, plot=False)
logger.log("test predictions stat:\n" + str(pred_stats))

logger.log(
    "single fold metrics:\n"+
    f"Best test_f1@best_epoch {model.val_f1_best_metric:.2f}, test_loss@best_epoch {model.val_loss_best_metric:.2f} at epoch {model.best_epoch}"
)
logger.log(f"test AUC: {auc_dict['AUC']:.2f} | test AUPR: {auc_dict['AUPR']:.2f}")

fig, ax = model.plot_confusion_matrix(
    preds,
    loaders["val_loader"].dataset.labels,
    save_path=f"{inference_dir}/test_confusion_matrix.png",
    title=f"Test confusion matrix {sweep_id}",
    plot=False,
)

test_results = trainer.test(model, loaders["val_loader"],
                            ckpt_path=inference_dir + f"/{ckpt_name}.ckpt",
                            verbose=1)[0]

# save the model config
if run_mode == "train":
    with open(f"{inference_dir}/features_config.json", "w") as f:
        try:
            json.dump(features_config, f)
        except Exception as e:
            print(e)

    with open(f"{inference_dir}/config.json", "w") as f:
        try:
            cls_config["class_weights"] = cls_config["class_weights"].tolist()
            cls_config["weights"] = chkpt_call.best_model_path.split("/")[-1]
            json.dump(cls_config, f)
        except Exception as e:
            print(e)

if wandb_log:
    aa_features_shape, ss_features_shape = return_aa_ss_shapes(loaders, use_aa, use_sf)
    hparams = {
        "Batch Size": BS,
        "Features": features_names,
        "Epochs": EPOCHS,
        "Features Dim": ((aa_features_shape[0], ss_features_shape[0]), FE_name),
        "Test Subset": TEST_SIZE,
        "Data Version": DATA_VERSION,
        "lr Scheduler": trainer.lr_scheduler_configs[0].name
        if trainer.lr_scheduler_configs
        else "Not Initialized",
        "CRISPR Combination Mode": CRISPR_MODE,
        "sweep_id": sweep_id,
        "use_sf": use_sf,
        "use_aa": use_aa,
        "Weight Address wandb": wb_logger.experiment.dir + "/best_model.ckpt",
        "Weights local": chkpt_call.best_model_path,
        "Train Samples": len(train_data),
        "Test Samples": len(test_data),
        "Exclude": "_".join(exclude_mode_dict[excl_mode])
        if exclude_mode_dict[excl_mode] is not None
        else None,
        "Monitor CKPT": monitor_ckpt,
        "Under Sampling": undersample,
    }
    wb_logger.log_hyperparams(hparams)
    wb_logger.log_metrics(pred_stats)

    best_res = {
        "test_f1": test_results["test_f1"],
        "test_acc": test_results["test_acc"],
        "test_loss": test_results["test_loss"],
        "test_AUC": auc_dict["AUC"],
        "test_AUPR": auc_dict["AUPR"],
    }
    wb_logger.log_metrics(best_res)
    logger.log("finishing up wandb logs...")
    wandb.log({"confusion_matrix": wandb.Image(fig)})

    wandb.finish()

if run_mode == "train":
    try:
        os.remove(f"{inference_dir}.zip", exist_ok=True)
        logger.log(f"zipping inference folder to  {inference_dir}.zip")
        shutil.make_archive(inference_dir, "zip", inference_dir)
    except:
        pass

logger.log("All Done!")