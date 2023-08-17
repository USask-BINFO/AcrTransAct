#!/usr/bin/env python
# coding: utf-8
import wandb
import os
import json
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from utils.data_utils import *
from utils.model_utils import *
from utils.ppi_utils import *
from utils.trainer import *
from utils.misc import *

from utils.logger import Logger
from datetime import datetime
from utils.common_vars import *
import shutil

os.environ["WANDB_SILENT"] = "true"
seed_everything(RANDOM_STATE)

time_stamp = datetime.now().strftime("%m-%d_%H:%M:%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ARGS
args = parse_arguments()
model_type = args.model_type
feature_mode = args.feature_mode
esm_version = args.esm_version
do_sweep = args.do_sweep 
wandb_log = args.wandb_log
cross_val = args.cross_val
undersample = args.undersample
excl_mode = args.excl_mode
optimize_metric = args.optimize_metric
monitor_ckpt = args.monitor_ckpt
run_mode = args.run_mode
sweep_id_ = args.sweep_id
label_smoothing = args.label_smoothing
print(args)

if run_mode =="eval":
    do_sweep = False
    wandb_log = False
    cross_val = False

exclude_mode_dict = {
    0: None,
    1: ["K12", "ATCC39006_IF"],
    2: ["K12", "PaLML1_DVT419"],
}
if monitor_ckpt.lower().__contains__("f1"):
    opt = "f1"
elif monitor_ckpt.lower().__contains__("loss"):
    opt = "loss"

use_sf, use_aa, features_names = setup_features(feature_mode, esm_version)

####################################
###     Data Preprocessing       ###
####################################

features_config = return_features_config(features_names, task="PPI")

features_config["exclude_mode_dict"] = exclude_mode_dict 
features_config["exclude_mode"] = excl_mode
features_config["use_aa"] = use_aa
features_config["use_sf"] = use_sf
features_config["features_names"] = features_names

logger = Logger(f"./code/logs/", f"logs_{time_stamp}_{model_type}.log")
logger.log(f"project name: {PROJ_VERSION}")
logger.log(f"reading data from: {INHIBITION_EXCEL_FILE}")
logger.log(f"debug mode: {DEBUG_MODE}")
logger.log(f"args: {args}")
logger.log(f"features names: {features_names}")

inhibition_df = pd.read_excel(INHIBITION_DF_PATH)
inhibition_df = inhibition_df[inhibition_df["use"] == 1]
inhibition_df = inhibition_df[inhibition_df["Acr_seq"].str.len() > 1]

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

features_aa, features_sf = load_extrated_features(feature_mode, features_config, data)

split_dict = split_data(
    inhibition_df, data, features_aa, features_sf, test_size=TEST_SIZE
)

train_df = split_dict["train_df"]
test_df = split_dict["test_df"]
train_data = split_dict["train_data"]
test_data = split_dict["test_data"]

plot_train_test_lbl_sys(
    train_df,
    test_df,
    save_path=f"{VERSION_FOLDER}/label_dist_crispr_sys.jpg",
    plot=0,
)

if use_aa:
    X_train_aa = split_dict["X_train_aa"]
    X_test_aa = split_dict["X_test_aa"]
if use_sf:
    X_train_sf = split_dict["X_train_sf"]
    X_test_sf = split_dict["X_test_sf"]

y_train = [d["inhibition"] for d in train_data]
y_test = [d["inhibition"] for d in test_data]

logger.log(f"Number of samples in train: {len(train_data)}")
logger.log(f"Number of samples in test: {len(test_data)}")

loaders, seq_len_aa, seq_len_sf = prepare_loaders(split_dict, use_aa, use_sf, device)

trainer = AcrTransActTrainer(data,
                             train_data,
                             split_dict,
                             features_config,
                             monitor_ckpt,
                             model_type,
                             time_stamp,
                             optimize_metric,
                             device
                             )
####################################
###  Hyperparameter Sweeping     ###
####################################
if do_sweep and sweep_id_ is None:
    sweep_id = trainer.start_sweep(SWEEP_RUNS)
else:
    # not currently working with the new data
    raise NotImplementedError
    # sweep_key = model_type + "_" + features_names
    # with open(SWEEP_SETTINGS_JSON, "r") as f:
    #     sweep_settings = json.load(f)
    # sweep_id = sweep_settings[sweep_key] if sweep_id_ is None else sweep_id_
    # logger.log(f"using configs from sweep_id: {sweep_id}, read from disk")

inference_dir =  f"./code/results/{PROJ_VERSION}/inf_{opt}_opt_{model_type}"+\
                 f"/{sweep_id if sweep_id is not None else 'no_sweep'}"
if exclude_mode_dict[excl_mode] is not None:
    inference_dir = inference_dir+f"_excl_{'_'.join(exclude_mode_dict[excl_mode])}"
else:
    inference_dir = inference_dir+"_no_excl"
inference_dir+= "_"+features_names+"/"
os.makedirs(inference_dir, exist_ok=True)
aa_features_shape, ss_features_shape = return_aa_ss_shapes(loaders, use_aa, use_sf)

if sweep_id_ is not None or do_sweep: # if sweep_id is in input args or sweeping is performed
    sweep_id = sweep_id_ if sweep_id_ is not None else sweep_id
    cls_config = load_best_cls_config(sweep_id, aa_features_shape, ss_features_shape, inference_dir)
else:
    with open(f"{inference_dir}/config.json", "r") as f:
        cls_config = json.load(f)

cls_model_name = return_PPI_name(cls_config, model_type)

save_cv_dir = f"./code/results/{PROJ_VERSION}/cv_rep_res/{sweep_id}_{time_stamp}/"
os.makedirs(save_cv_dir, exist_ok=True)
logger.log(f"Cross-val dir: {save_cv_dir}")
logger.log(f"inference dir: {inference_dir}")

####################################
###   Cross Validation Training  ###
####################################
if wandb_log:
    logger.log("WandB Logging started ...")
    wb_logger = WandbLogger(
        project=PROJ_VERSION,
        name=cls_model_name,
        log_model=False,
    )

if cross_val:

    cv_rep_res = trainer.cross_validation(cv_config=cls_config, reps=1, folds=5, epochs=100) 
    avg_results_cv = return_avg_cv_results(cv_rep_res)

    if wandb_log:
        wandb.log(
            {"cv_folds": CV_FOLDS,
            "Kfold_repeats": REPEAT_CV,
            }
        )
        wandb.log(avg_results_cv)
        with open(f"{save_cv_dir}/cv_config.json", "w") as f:
            json.dump(cv_rep_res, f)

####################################
### Train and Eval for inference ###
####################################
cls_config["class_weights"] = compute_class_weights(y_train, device)

if label_smoothing>0:
    smooth_labels = [label_smoothing if y == 0.
                      else 1-label_smoothing for y in loaders["train_loader"].dataset.labels]
    loaders["train_loader"].dataset.labels = torch.tensor(smooth_labels)
    
    
if run_mode == "eval" and sweep_id_ is None:
    ckpt_name = cls_config["weights"].replace(".ckpt", "")
else:
    ckpt_name = "best_model"+"_"+time_stamp

# train on train-val set
if run_mode == "train":
    model, test_results = trainer.train_eval_model(cls_config, loaders, inference_dir, ckpt_name, epochs=250)
    logger.log(f"checkpoint at: {inference_dir+ckpt_name}")
    logger.log(f"Number of trainable parameters: {count_parameters(model)}")
    
    with open(f"{inference_dir}/features_config.json", "w") as f:
        json.dump(features_config, f)
    with open(f"{inference_dir}/config.json", "w") as f:
        cls_config["weights"] = ckpt_name
        cls_config["class_weights"] = cls_config["class_weights"].tolist()
        json.dump(cls_config, f)

else:
    test_results = trainer.eval_model(cls_config, inference_dir, ckpt_name, loaders["val_loader"])

preds = model.pred_val(
    loaders["val_loader"],
    return_probs=True,
    use_aa=use_aa,
    use_sf=use_sf,
)
preds_binary = [1. if p > 0.5 else 0. for p in preds]

auc_dict = return_AUC(
    preds,
    y_test,
    save_path=inference_dir,
    plot_ROC=False,
    title=f"{model_type} {features_names}"
)
pred_stats = prediction_stats(preds_binary, test_data, plot=False)
logger.log("test predictions stat:\n" + str(pred_stats))
logger.log(f"test AUC: {auc_dict['AUC']:.2f} | test AUPR: {auc_dict['AUPR']:.2f}")

fig, ax = trainer.plot_confusion_matrix(
    preds_binary,
    loaders["val_loader"].dataset.labels,
    save_path=f"{inference_dir}/conf_mat_{time_stamp}.png",
    title=f"{model_type} {features_names}",
    plot=False
)

if wandb_log:
    hparams = {
        "Batch Size": BS,
        "Features": features_names,
        "Epochs": EPOCHS,
        "Features Dim": ((aa_features_shape[0], ss_features_shape[0]), features_names),
        "Test Subset": TEST_SIZE,
        "CRISPR Combination Mode": CRISPR_MODE,
        "sweep_id": sweep_id,
        "use_sf": use_sf,
        "use_aa": use_aa,
        "Weight Address wandb": wb_logger.experiment.dir + "/best_model.ckpt",
        "Weights local": inference_dir+ckpt_name,
        "Train Samples": len(train_data),
        "Test Samples": len(test_data),
        "Exclude": "_".join(exclude_mode_dict[excl_mode])
        if exclude_mode_dict[excl_mode] is not None
        else None,
        "Monitor CKPT": monitor_ckpt,
        "Under Sampling": undersample,
        "Label Smoothing": label_smoothing,
        "Time Stamp": time_stamp,
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