import torch
import numpy as np
from utils.data_utils import one_hot_enc, read_fasta
from utils.ss_utils import *
import wandb
from utils.common_vars import *
from utils.model_utils import get_best_run_wandb


def return_target_Cas(path, proteins):
    """
    This function returns a list of Cas (Csy, Cse, Cas) protein sequences from a fasta file.
    The Cas protein names are given in the proteins list.

    Parameters
    ----------
    path : str
        path to fasta file
    proteins : list
        list of cas protein names from the csv file for each row
    """

    cas_seqs = []
    seqs = read_fasta(path, return_desc=True)

    for k in seqs.keys():
        for p in proteins:
            if k.replace("/", " ").find(p) != -1:
                cas_seqs.append(seqs[k])

    assert len(cas_seqs) == len(
        proteins
    ), f"There should be {len(proteins)} Cas proteins in the file {path} but found {len(cas_seqs)}."

    return cas_seqs


def read_ppi_data_df(
    inhibition_df,
    CRISPR_df,
    Cas_proteins_col="possible_Cas_targets",
    CRISPR_system_col="CRISPR_name_short",
    verbose=0,
):
    """
    This function reads the inhibition excel file and returns a list of dictionaries.
    Each dictionary contains the following keys: Acr_name, Acr, Cas_proteins, inhibition

    Parameters
    ----------
    inhibition_df : pandas dataframe
        dataframe with the Acr and CRISPR inhibition information
    CRISPR_df : pandas dataframe
        dataframe with the Cas sequences
    cas_proteins_column : str, optional
        the name of the column in the inhibition_df that contains the Cas proteins
    verbose : int, optional
        verbose level, by default 0

    Returns
    -------
    list
        list of dictionaries with the following keys: Acr_name, CRISPR_system, Acr, Cas_proteins, and inhibition
    """
    data = []
    for _, row in inhibition_df.iterrows():
        if row[Cas_proteins_col] is not np.nan:
            cas_list, cas_ids, cas_names = [], [], []
            cr_system = row[CRISPR_system_col]

            # iterate inside the CRISPR_df and find the Cas proteins for the current CRISPR system
            # if the Cas protein is found, append the Cas sequence to the Cas_list
            cas_order = []  # for ordering of the Cas proteins
            for _, cr_row in CRISPR_df.iterrows():
                if cr_row["system"] == cr_system:
                    if int(cr_row["used_in_inhibition"]) == 1:
                        cas_list.append(cr_row["seq"])
                        cas_ids.append(cr_row["id"])
                        cas_names.append(cr_row["Cas_name"])
                        cas_order.append(cr_row["order_in_features"])

            # sort cas_list, cas_ids based on cas_order
            cas_list = [x for _, x in sorted(zip(cas_order, cas_list))]
            cas_ids = [x for _, x in sorted(zip(cas_order, cas_ids))]
            cas_names = [x for _, x in sorted(zip(cas_order, cas_names))]

            known_Cas_target = (
                row["target_Cas_protein"]
                if any(char.isdigit() for char in str(row["target_Cas_protein"]))
                else "unknown"
            )

            acr_cas_dic = {
                "inhibition": row["inhibition"],
                "Acr_name": f'{row["Acr_family"]}_{row["phage_species"]}',
                "Acr_id": row["Acr_id"],
                "Acr_seq": row["Acr_seq"],
                "CRISPR_system": row[CRISPR_system_col],
                "CRISPR_system_type" : row["inhibition_type"],
                "Cas_names": cas_names,
                "Cas_proteins": cas_list,
                "Cas_ids": cas_ids,
                "known_Cas_target": known_Cas_target,
            }

            data.append(acr_cas_dic)

            if verbose:
                print(f"Pair exctracted {row['Acr_family']}, {row[CRISPR_system_col]}")
        else:
            print(f"Pair skipped {row['Acr_family']}, {row[CRISPR_system_col]}")
    return data


def read_crispr_data_df(
    CRISPR_df,
    verbose=0,
):
    """
    This function reads the CRISPR dataframe and returns a list of dictionaries, each dic is for one CRISPR system.
    Each dictionary contains the following keys: CRISPR_system, Cas_proteins, Cas_ids, Cas_names

    Parameters
    ----------
    CRISPR_df : pandas dataframe
        dataframe with the Cas sequences
    verbose : int, optional
        verbose level, by default 0

    Returns
    -------
    data : list
        list of dictionaries with the following keys: CRISPR_system, Cas_proteins, Cas_ids, Cas_names
    """

    data = []
    systems = set(CRISPR_df["system"].tolist())

    # NOTE: the Cas proteins should have the same ordering as they have in the inhibition excel file!
    # this is enforced by the order_in_features column in the CRISPR_df

    for system in systems:
        cas_seqs, cas_ids, cas_names, cas_orders = [], [], [], []
        for _, row in CRISPR_df.iterrows():
            if row["system"] == system and row["used_in_inhibition"] == 1:
                cas_seqs.append(row["seq"])
                cas_ids.append(row["id"])
                cas_names.append(row["Cas_name"])
                cas_orders.append(row["order_in_features"])

        # sort the Cas proteins based on their order in the inhibition excel file
        cas_seqs = [x for _, x in sorted(zip(cas_orders, cas_seqs))]
        cas_ids = [x for _, x in sorted(zip(cas_orders, cas_ids))]
        cas_names = [x for _, x in sorted(zip(cas_orders, cas_names))]

        Cas_data = {
            "CRISPR_system": system,
            "Cas_proteins": cas_seqs,
            "Cas_ids": cas_ids,
            "Cas_names": cas_names,
        }

        data.append(Cas_data)
        if verbose:
            print(f"Seqs exctracted {system}")

    return data


def return_config_by_run(run_id, proj):
    """
    This function returns the config and summary of
    the model from a run that is recorded by wandb.
    """
    run = wandb.Api().run(f"moeinh77/{proj}/{run_id}")
    cls_config = {}
    try:
        for key in run.config.keys():
            cls_config[key.replace("params/", "")] = run.config[key]
    except Exception as e:
        print(e)

    cls_config["class_weights"] = torch.tensor(
        cls_config["class_weights"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float,
    )

    return cls_config


def return_PPI_name(PPI_config, model_type):
    """
    This function returns the name of the PPI model

    Parameters:
    ----------------
    PPI_config (dict):
        dictionary containing the configuration parameters.
    model_type (str):
        the type of the model (CNN or RNN)

    Returns:
    ----------------
    name (str):
        the name of the PPI model
    """
    name = model_type + "_"
    if model_type == "CNN":
        name += f"K{PPI_config['kernel_size_aa']}_" + f"{PPI_config['kernel_size_sf']}_"

    name += (
        f"CH{PPI_config['out_channels_aa']}_"
        + f"{PPI_config['out_channels_sf']}_"
        + f"FC{PPI_config['FC_nodes']}_"
        + f"M{PPI_config['mode']}_"
        + f"L{PPI_config['num_main_layers']}"
    )
    return name


def return_features_config(FE_name, task):
    if task == "PPI":
        output_attentions = False
        add_cross_attention = False
        output_hidden_states = True
    else:
        return NotImplementedError()

    return {
        "model_name": FE_name,
        "task": task,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_attentions": output_attentions,
        "add_cross_attention": add_cross_attention,
        "output_hidden_states": output_hidden_states,
        "CRISPR_mode": CRISPR_MODE,
        "Acr_ss_df": str(VERSION_FOLDER / "Acr" / f"Acrs_netsurfp.xlsx"),
        "CRISPR_ss_df": str(VERSION_FOLDER / "CRISPR" / f"CRISPR_netsurfp.xlsx"),
        "Acr_Cas_mode": 2,  # 2 for both Acr and Cas features, 1 for Cas only
        "channels_first": True,
        "weights_dir": f"Rostlab/{FE_name}" if FE_name.find("prot") != -1 else None,
        "random_state": RANDOM_STATE,
    }


def get_avg_std_rep(CV_models, logger=None):
    """averages all the folds of a rep and returns the average for each rep.
    Also logs the average and std of each rep."""
    rep_avg_f1 = []
    rep_avg_auc = []
    rep_avg_acc = []
    rep_avg_aupr = []
    rep_avg_prec = []
    rep_avg_prec = []
    rep_avg_recall = []
    rep_avg_loss = []

    for i, rep in enumerate(CV_models): # for each rep

        folds_val_acc = [fold["accuracy"] for fold in rep]
        folds_val_f1 = [fold["f1"] for fold in rep]
        folds_val_loss = [fold["loss"] for fold in rep]
        folds_val_auc = [fold["auc"] for fold in rep]
        folds_val_aupr = [fold["aupr"] for fold in rep]
        folds_val_prec = [fold["precision"] for fold in rep]
        folds_val_recall = [fold["recall"] for fold in rep]

        avg_val_acc = np.mean(folds_val_acc)
        avg_val_f1 = np.mean(folds_val_f1)
        avg_val_loss = np.mean(folds_val_loss)
        std_val_acc = np.std(folds_val_acc)
        std_val_f1 = np.std(folds_val_f1)
        std_val_loss = np.std(folds_val_loss)
        avg_val_auc = np.mean(folds_val_auc)
        avg_val_aupr = np.mean(folds_val_aupr)
        std_val_auc = np.std(folds_val_auc)
        std_val_aupr = np.std(folds_val_aupr)
        avg_val_prec = np.mean(folds_val_prec)
        avg_val_recall = np.mean(folds_val_recall)

        if logger is not None:
            logger.log(f"rep {i+1}:")
            logger.log(f"avg_val_acc: {avg_val_acc:.2f} +- {std_val_acc:.2f}")
            logger.log(f"avg_val_f1: {avg_val_f1:.2f} +- {std_val_f1:.2f}")
            logger.log(f"avg_val_loss: {avg_val_loss:.2f} +- {std_val_loss:.2f}")
            logger.log(f"avg_val_auc: {avg_val_auc:.2f} +- {std_val_auc:.2f}")
            logger.log(f"avg_val_aupr: {avg_val_aupr:.2f} +- {std_val_aupr:.2f}")
            logger.log(f"avg_val_prec: {avg_val_prec:.2f}")
            logger.log(f"avg_val_recall: {avg_val_recall:.2f}")

        rep_avg_f1.append(round(avg_val_f1, 2))
        rep_avg_auc.append(round(avg_val_auc, 2))
        rep_avg_acc.append(round(avg_val_acc, 2))
        rep_avg_aupr.append(round(avg_val_aupr, 2))
        rep_avg_prec.append(round(avg_val_prec, 2))
        rep_avg_recall.append(round(avg_val_recall, 2))
        rep_avg_loss.append(round(avg_val_loss, 2))

    return {
        "rep_avg_f1": rep_avg_f1,
        "rep_avg_auc": rep_avg_auc,
        "rep_avg_acc": rep_avg_acc,
        "rep_avg_aupr": rep_avg_aupr,
        "rep_avg_prec": rep_avg_prec,
        "rep_avg_recall": rep_avg_recall,
        "rep_avg_loss": rep_avg_loss,
    }


def return_avg_cv_results(reps_results, logger=None):
    """averages the results of all reps and returns the average for each metric."""

    reps_stats= get_avg_std_rep(reps_results, logger)

    return {
        "F1_CV": np.mean(reps_stats['rep_avg_f1']),
        #np.round(np.mean([res["f1"] for res in results_cv]), 2),
        "Acc_CV": np.mean(reps_stats['rep_avg_acc']),
        #np.round(np.mean([res["accuracy"] for res in results_cv]), 2),
        "Loss_CV": np.mean(reps_stats['rep_avg_loss']),
          #np.round(np.mean([res["loss"] for res in results_cv]), 2),
        "AUC_CV": np.mean(reps_stats['rep_avg_auc']),
          #np.round(np.mean([res["auc"] for res in results_cv]), 2),
        "AUPR_CV": np.mean(reps_stats['rep_avg_aupr']),
          #np.round(np.mean([res["aupr"] for res in results_cv]), 2),
        "Precision_CV": np.mean(reps_stats['rep_avg_prec']),
          #np.round(np.mean([res["precision"] for res in results_cv]), 2),
        "Recall_CV": np.mean(reps_stats['rep_avg_recall']),
        #np.round(np.mean([res["recall"] for res in results_cv]), 2),
        "f1_each_rep_CV": reps_stats['rep_avg_f1'],
        "auc_each_rep_CV": reps_stats['rep_avg_auc'],
        "acc_each_rep_CV": reps_stats['rep_avg_acc'],
        "aupr_each_rep_CV": reps_stats['rep_avg_aupr'],
        "prec_each_rep_CV": reps_stats['rep_avg_prec'],
        "recall_each_rep_CV": reps_stats['rep_avg_recall'],
    }


def each_CRISPR_results(CV_models):
    """
    Returns a dictionary with the predictions details of each fold in each rep for each bacteria
    """
    rep_res = {}
    for i, rep in enumerate(CV_models): # for each rep
        cv_res = {}
        fold = 1
        for cv in rep: # for each fold
            dict_count = {}
            for bacteria in cv["preds_stats"]:
                if bacteria not in dict_count:
                    dict_count[bacteria] = {
                        "correct": 0,
                        "wrong": 0,
                        "Accuracy": 0
                    }
                dict_count[bacteria]["correct"] += cv["preds_stats"][bacteria]["correct"]
                dict_count[bacteria]["wrong"] += cv["preds_stats"][bacteria]["wrong"]
                dict_count[bacteria]["Accuracy"] = round(
                    dict_count[bacteria]["correct"] / (dict_count[bacteria]["correct"] + dict_count[bacteria]["wrong"]), 2
                )
            
            cv_res[f"fold {fold}"] = dict_count
            fold += 1
        rep_res[f"rep {i+1}"] = cv_res

    return rep_res


def avg_each_CRISPR_results(rep_res):
    system_acc = {}
    for rep in rep_res.keys():
        for fold in rep_res[rep].keys():
            for system in rep_res[rep][fold].keys():
                if system != 'Accuracy':
                    continue
                if system not in system_acc:
                    system_acc[system] = []
                system_acc[system].append(rep_res[rep][fold][system])

    avg_acc = {}

    for system, acc_list in system_acc.items():
        avg_acc[system] = sum(acc_list) / len(acc_list)

    return avg_acc


def return_cls_config(config):
    """
    This function returns the config of the classifier dictionary from the config file of the sweep

    Parameters
    ----------
    config : dict
        config file of the sweep

    Returns
    -------
    dict
        PPI config dictionary
    """

    return {
        "lr": config["lr"],
        "reduce_lr": REDUCE_LR_TECHNIQUE,
        "monitor_metric_lr": MONITOR_LR,
        "kernel_size_aa": config["kernel_size_aa"] if config["use_aa"] else 0,
        "kernel_size_sf": config["kernel_size_sf"] if config["use_sf"] else 0,
        "out_channels_aa": config["out_channels_esm"] if "out_channels_esm" in config else config["out_channels_aa"],
        "out_channels_sf": config["out_channels_sf"] if config["use_sf"] else 0,
        "weight_decay": config["weight_decay"],
        "dout": config["dout"],
        "mode": config["mode"],
        "FC_nodes": config["FC_nodes"],
        "optimizer": config["optimizer"],
        "use_sf": config["use_sf"],
        "use_aa": config["use_aa"] if "use_aa" in config else True, # some older sweeps don't have this parameter
        # "mode_cyclic": config["mode_cyclic"] if config["reduce_lr"] == "cyclic" else None,
        # "base_lr": config["base_lr"] if config["reduce_lr"]== "cyclic" else 0,
        # "max_lr": config["max_lr"] if config["reduce_lr"]== "cyclic" else 0,
        # "step_size": config["step_size"] if config["reduce_lr"]== "cyclic" else 0,
        "lr_reduce_factor": REDUCE_LR_FACTOR,
        "class_weights": config["class_weights"] if "class_weights" in config else None,
        "num_main_layers": config["num_main_layers"] if "num_main_layers" in config else 1,
   }


def load_best_cls_config(sweep_id, aa_features_shape, ss_features_shape, inference_dir):
    """
    Loads the best cls config from the sweep hisotry and saves it in the inference dir
    """
    best_run = get_best_run_wandb(sweep_id, PROJ_VERSION)
    cls_config = return_cls_config(best_run)
    cls_config["seq_len_aa"] = aa_features_shape[1] 
    cls_config["seq_len_sf"] = ss_features_shape[1] 
    cls_config["hidden_size_aa"] = aa_features_shape[0] 
    cls_config["hidden_size_sf"] = ss_features_shape[0]
    cls_config["optimize_metric"] = OPTIMIZE_METRIC
    cls_config["monitor_metric_lr"] = MONITOR_LR
    cls_config["sweep_id"] = sweep_id

    with open(f"{inference_dir}/config.json", "w") as f: # save the sweep results
        json.dump(cls_config, f) 

    return cls_config


def get_config_from_pre_tune(config, monitor_ckpt, model_type):
    """loads the config from the tuning config before the search has begun.
    used in the search loop."""
    return {
    "monitor_metric_lr": MONITOR_LR,
    "optimize_metric": monitor_ckpt,
    "kernel_size_aa": config.kernel_size_aa if config.use_aa else 0,
    "kernel_size_sf": config.kernel_size_sf if config.use_sf else 0,
    "out_channels_aa": config.out_channels_aa if config.use_aa else 0,
    "out_channels_sf": config.out_channels_sf if config.use_sf else 0,
    "weight_decay": config.weight_decay,
    "dout": config.dout,
    "mode": config.mode,
    "FC_nodes": config.FC_nodes,
    "lr": config.lr,
    "reduce_lr": REDUCE_LR_TECHNIQUE,
    "optimizer": config.optimizer,
    "use_sf": config.use_sf,
    "use_aa": config.use_aa,
    # "max_lr": config.max_lr,
    # "step_size": config.step_size,
    "lr_reduce_factor": REDUCE_LR_FACTOR,
    "num_main_layers": config.num_main_layers if "num_main_layers" in config else 1,
    "channels_first": True,
    "random_state": RANDOM_STATE,
}