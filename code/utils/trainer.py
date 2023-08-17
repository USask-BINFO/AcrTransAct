import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn.metrics import confusion_matrix
import time
import os
from utils.common_vars import *
from utils.data_utils import (
    return_aa_ss_shapes,
    return_AUC,
    return_loaders,
    compute_class_weights
)
from utils.models.cnn_model import AcrTransAct_CNN
from utils.models.lstm_model import AcrTransAct_LSTM
from utils.misc import free_mem
from utils.logger import Logger
from utils.ppi_utils import (
    get_config_from_pre_tune,
    return_PPI_name,
    return_avg_cv_results,
)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import random
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt

class AcrTransActTrainer:
    def __init__(
        self,
        data,
        train_data,
        split_dict,
        features_config,
        monitor_ckpt,
        model_type,
        time_stamp,
        optimize_metric,
        device
    ):
        self.data = data
        self.train_data = train_data
        self.split_dict = split_dict
        self.features_config = features_config
        self.monitor_ckpt = monitor_ckpt
        self.model_type = model_type
        self.time_stamp = time_stamp
        self.use_aa = features_config["use_aa"]
        self.use_sf = features_config["use_sf"]
        self.features_names = features_config["features_names"]
        self.model_type_dic = {
            "CNN": AcrTransAct_CNN,
            "LSTM": AcrTransAct_LSTM,
        }
        self.optimize_metric = optimize_metric
        self.logger = Logger(
            f"./code/logs/", f"logs_{self.time_stamp}_{self.model_type}.log"
        )
        self.device = device

        self.cls_model_name = None
        self.sweep_id = None


    def cross_validation(self, cv_config, reps=1, folds=5, epochs=100, tune=False):
        """
        *
         tune: whether this function is being used whithin the wandb sweep search
        """
        ckpt_dir_cv = f"./code/results/{PROJ_VERSION}/temp_chkpts/"

        self.logger.log(f"checkpoint dir cv: {ckpt_dir_cv}")

        X_train_aa, X_train_sf = (
            self.split_dict["X_train_aa"],
            self.split_dict["X_train_sf"],
        )
        use_aa, use_sf = self.features_config["use_aa"], self.features_config["use_sf"]
        exclude_mode_dict = self.features_config["exclude_mode_dict"]
        excl_mode = self.features_config["exclude_mode"]

        kf_stratified = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=RANDOM_STATE
        )
        labels_CRISPR_sys = [
            str(d["inhibition"]) + "_" + d["CRISPR_system"] for d in self.train_data
        ]
        cv_rep_res = []
        labels = [d["inhibition"] for d in self.train_data]
        if not tune:
            self.logger.log("Cross validation started...")

        for rep in range(reps):
            cv_folds_res = []
            for fold, (train_index, test_index) in enumerate(
                kf_stratified.split(
                    X=X_train_aa if use_aa else X_train_sf, y=labels_CRISPR_sys
                )
            ):
                if not tune and not DEBUG_MODE:
                    print(
                        f"> ****** Fold {fold+1} of {folds} | Rep: {rep+1}/{reps} ******"
                    )
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

                cv_config["class_weights"] = compute_class_weights(y_train, device=self.device)

                train_val_data = {
                    "X_train_aa": X_train_aa_fold if use_aa else None,
                    "X_val_aa": X_val_aa_fold if use_aa else None,
                    "X_train_sf": X_train_sf_fold if use_sf else None,
                    "X_val_sf": X_val_sf_fold if use_sf else None,
                    "y_train": y_train,
                    "y_val": y_val,
                }

                try:
                    loaders = return_loaders(
                    train_val_data, bs=BS, use_sf=use_sf, use_aa=use_aa, drop_last=True
                    )
                except Exception as e:
                    self.logger.log(f"!!! Error in CV: {e}")

                aa_features_shape, ss_features_shape = return_aa_ss_shapes(
                    loaders, use_aa, use_sf
                )
                ###################
                ## MODEL CONFIGS ##
                ###################
                i = 1 if self.features_config["channels_first"] else 0
                cv_config["seq_len_aa"] = aa_features_shape[1] if use_aa else 0
                cv_config["seq_len_sf"] = ss_features_shape[1] if use_sf else 0
                cv_config["hidden_size_aa"] = (
                    aa_features_shape[0] if use_aa else 0
                )
                cv_config["hidden_size_sf"] = (
                    ss_features_shape[0] if use_sf else 0
                )
                if tune:
                    ckpt_dir_cv = f"./code/results/{PROJ_VERSION}/temp_chkpts_tune/"

                chkpt_dir = (
                    f"{ckpt_dir_cv}/{self.cls_model_name}_{fold_name}_{self.time_stamp}/"
                )

                if exclude_mode_dict[excl_mode] is not None:
                    chkpt_dir = (
                        chkpt_dir[:-1]
                        + f"_excl{'_'.join(exclude_mode_dict[excl_mode])}"
                    )
                else:
                    chkpt_dir = chkpt_dir[:-1] + f"_no_excl/"

                os.makedirs(chkpt_dir, exist_ok=True)
                chkpt_name = "best_model"

                if os.path.isfile(chkpt_dir + chkpt_name):
                    chkpt_dir = chkpt_dir[:-1] + str(random.randint(0, 1000)) + "/"

                # train model # 
                model, test_results = self.train_eval_model(cv_config, loaders, chkpt_dir, chkpt_name, epochs=epochs)

                print(len(loaders["val_loader"].dataset.labels))

                preds = model.pred_val(
                    loaders["val_loader"], use_aa, use_sf, return_probs=True
                )
                auc_dict = return_AUC(
                    preds,
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

                if not tune:
                    print(f"> RESULTS: {pred_results}, BEST EPOCH: {model.best_epoch}")
                    self.logger.log(
                        f"fold {fold+1}/{CV_FOLDS} rep {rep+1}/{REPEAT_CV} results:\n{pred_results}, best epoch: {model.best_epoch}"
                    )
                ######################
                ## LOGGING RESULTS ##
                ######################
                # fold results for when retriving each fold
                cv_fold_res = {
                    "Fold": fold + 1,
                    "Rep": rep + 1,
                    "seq_len_aa": cv_config["seq_len_aa"],
                    "seq_len_sf": cv_config["seq_len_sf"],
                    "check_point": f"{chkpt_dir}/{chkpt_name}.ckpt",
                }
                # adding the info in pred_results to cv_fold_res
                for key in pred_results.keys():
                    cv_fold_res[key] = pred_results[key]

                cv_folds_res.append(cv_fold_res)  # fold results

                del model
                del loaders
                free_mem()

            cv_rep_res.append(cv_folds_res)  # rep results

        return cv_rep_res


    def train_eval_model(self, cls_config, loaders, chkpt_dir, ckpt_name, epochs=250):
        
        ckpt_path = chkpt_dir+ckpt_name+".ckpt"

        chkpt_call = ModelCheckpoint(
                    dirpath=chkpt_dir,
                    filename=ckpt_name,
                    save_top_k=1,
                    monitor=self.monitor_ckpt,
                    mode="min" if self.monitor_ckpt.__contains__("loss") else "max",
                    save_weights_only=True,
                )

        es_call = EarlyStopping(
            monitor=self.monitor_ckpt if not DEBUG_MODE else "train_loss",
            patience=30,
            mode="min" if self.monitor_ckpt.__contains__("loss") else "max",
            verbose=0,
            min_delta=0.005,
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_progress_bar=True,
            enable_model_summary=False,
            accelerator="gpu",
            devices=1,
            callbacks=[chkpt_call, es_call],
            max_steps=1 if DEBUG_MODE else -1,
        )

        model = self.model_type_dic[self.model_type](
            cls_config,
            channels_first=True,
        )

        trainer.fit(
            model,
            loaders["train_loader"],
            loaders["val_loader"],
        )
        
        model.plot_history(running_mean_window=1,
                            save_path=f"{chkpt_dir}/{self.time_stamp}_history.png",
                            plot=False,
                            )

        model = self.model_type_dic[self.model_type].load_from_checkpoint(
                    ckpt_path,
                    cls_config,
                    channels_first=True,
                )
        
        model.eval()
        test_results = trainer.test(
                    model,
                    loaders["val_loader"],
                    ckpt_path=ckpt_path if not DEBUG_MODE else None,
                    verbose=0,
                )[0]
        
        return model, test_results


    def eval_model(self, cls_config, chkpt_dir, ckpt_name, val_loader, trainer):
        ckpt_path = chkpt_dir+ckpt_name+".ckpt"

        model = self.model_type_dic[self.model_type].load_from_checkpoint(
                        ckpt_path,
                        cls_config,
                        channels_first=self.features_config["channels_first"],
                    )
            
        model.eval()
        test_results = trainer.test(
                    model,
                    val_loader,
                    ckpt_path=ckpt_path if not DEBUG_MODE else None,
                    verbose=0,
                )[0]
        
        return test_results


    def _return_search_space(self):
        return {
            "name": f"{self.model_type}_{self.features_names}_{self.time_stamp}",
            "method": "bayes",
            "metric": {
                "name": self.optimize_metric,
                "goal": "maximize"
                if self.optimize_metric.lower().__contains__("f1")
                else "minimize",
            },
            "parameters": {
                "lr": {"min": 3e-4, "max": 3e-3},
                "weight_decay": {"min": 1e-3, "max": 1e-2},
                "dout": {"min": 0.3, "max": 0.5},
                "kernel_size_aa": {"values": [7] if self.use_aa else [0]},
                "kernel_size_sf": {"values": [7] if self.use_sf else [0]},
                "out_channels_sf": {"values": [4, 8] if self.use_sf else [0]},
                "out_channels_aa": {"values": [4, 8] if self.use_aa else [0]},
                "batch_size": {"values": [BS]},
                "FC_nodes": {"values": [4, 8]},
                "mode": {"values": [2]},
                "optimizer": {"values": ["Adam"]},
                "use_sf": {"values": [self.use_sf]},
                "use_aa": {"values": [self.use_aa]},
                "num_main_layers": {"values": [1, 2]},
            },
        }

    
    def _hparam_search(self):
        wandb.init(project=PROJ_VERSION)
        cls_config_tuning = get_config_from_pre_tune(
            wandb.config, self.monitor_ckpt, self.model_type
        )
        wb_logger = WandbLogger(log_model=False)

        cls_model_name = return_PPI_name(cls_config_tuning, self.model_type)

        hparams = {
            "Model Name": cls_model_name,
            "Feature Extractor": self.features_config["model_name"],
        }
        wb_logger.log_hyperparams(hparams)
        
        # perform cross validation
        cv_rep_res = self.cross_validation(
            reps=1,
            folds=5,
            cv_config=cls_config_tuning,
        )

        avg_results_cv = return_avg_cv_results(cv_rep_res)
        wb_logger.log_metrics(avg_results_cv)
        wandb.finish()
        free_mem()


    def start_sweep(self, sweep_runs=20):
        sweep_config = self._return_search_space()

        self.logger.log(
            f"starting the search, optimizing for: {self.optimize_metric}"
        )

        sweep_id = wandb.sweep(sweep_config, project=PROJ_VERSION)
        start = time.time()
        wandb.agent(
            sweep_id, self._hparam_search, count=sweep_runs if not DEBUG_MODE else 1
        )
        end = time.time()
        self.logger.log(f"total search time: {(end-start)/60:.2f} minutes, sweep_id:{sweep_id}")

        self.sweep_id = sweep_id

        return sweep_id
    

    def plot_confusion_matrix(
        self,
        preds,
        val_labels,
        save_path=None,
        title = None,
        cmap=plt.cm.Blues,
        plot=False
    ):
        """
        Plot confusion matrix

        Parameters
        ----------
        preds : numpy array
            predictions on validation set (either classes or probabilities)
        save_path : str, optional
            Path to save the plot, by default None
        cmap : matplotlib colormap, optional
            Colormap to use, by default plt.cm.Blues
        """
        labels = ["0", "1"]
        # preds = np.argmax(preds, axis=1)
        val_labels = val_labels.cpu().numpy()
        cm = confusion_matrix(val_labels, preds)

        fig, ax = plt.subplots(figsize=(4, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),  # Display count for each section
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 0.5 * np.max(cm) else "black",
                )
        fig.tight_layout()
        if title is not None:
            plt.title(title)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if plot:
            plt.show()

        return fig, ax
