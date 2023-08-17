import re
import torch
from Bio import SeqIO
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from utils.common_vars import *
from utils.ss_utils import return_ss_pt
from sklearn.model_selection import train_test_split
from utils.model_utils import comp_hidden_states
from utils.model_utils import load_feature_extractor
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import pickle
import os


class AA_SS_Dataset(Dataset):

    """
    This class is used to create a dataset for the ESM features and secondary structure features.
    """

    def __init__(self, encodings_esm, encodings_ss, labels=None, info=None):
        self.encodings_esm = encodings_esm
        self.encodings_ss = encodings_ss
        self.labels = labels
        self.info = info

    def __getitem__(self, idx):
        item_esm = self.encodings_esm[idx]
        item_ss = self.encodings_ss[idx]

        if self.labels is not None:
            label = self.labels[idx]
            item = ((item_esm, item_ss), label)

        else:
            item = (item_esm, item_ss)

        return item

    def __len__(self):
        return len(self.encodings_esm)


class AA_Dataset(Dataset):
    """
    This class is used to create a dataset for the ESM features.
    """

    def __init__(self, encodings_esm, labels=None, info=None):
        self.encodings_esm = encodings_esm
        self.labels = labels
        self.info = info

    def __getitem__(self, idx):
        item = self.encodings_esm[idx]
        if self.labels is not None:
            label = self.labels[idx]
            item = (item, label)

        return item

    def __len__(self):
        return len(self.encodings_esm)


def read_fasta(path, return_desc=False):
    """
    This function reads the fasta file at path address and returns
    a dictionary with ids as keys and sequences as values

    Parameters
    ----------
    path : str
        Path to fasta file
    return_desc : bool
        If True, the description of the fasta file will be used as the key. If False, the id will be used as the key.

    Returns
    -------
    seqs : dict
        Dictionary with ids as keys and sequences as values
    """

    seqs = {}
    with open(path) as fasta:
        fasta_sequences = SeqIO.parse(fasta, "fasta")

    for fasta in fasta_sequences:
        if return_desc:
            id = fasta.description
        else:
            id = fasta.id

        seqs[id] = str(fasta.seq)

    return seqs


def pad_features(X, max_len, channels_first=True):
    """
    Pads the input sequences to have the same length.
    Maximum length of the pad should come from the training set.

    Parameters
    ----------
    X: list
        List of sequences to be padded
    max_len: int
        Maximum length of the sequences
    channels_first: bool
        If True, the input sequences have the shape (batch_size, channels, length).

    Returns
    -------
    padded_X: torch.Tensor
        Padded sequences
    """
    # print("max_len: ", max_len)

    if channels_first:
        padded_X = [
            torch.nn.functional.pad(x, (0, max_len - x.size()[2], 0, 0)) for x in X
        ]
    else:
        padded_X = [
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.size()[1])) for x in X
        ]

    # print([x.size() for x in padded_X])
    padded_X = torch.stack(padded_X, dim=0)

    padded_X = padded_X.squeeze(1).float()
    # print("padded_X.size(), ", padded_X.size())

    return padded_X


def one_hot_enc(seq):
    """
    One-hot encodes the input sequence

    Parameters
    ----------
    seq : str
        Input sequence

    Returns
    -------
    one_hot_seq : torch.Tensor
        One-hot encoded sequence
    """
    aa_to_num = {
        "A": 0,
        "R": 1,
        "N": 2,
        "D": 3,
        "C": 4,
        "Q": 5,
        "E": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "L": 10,
        "K": 11,
        "M": 12,
        "F": 13,
        "P": 14,
        "S": 15,
        "T": 16,
        "W": 17,
        "Y": 18,
        "V": 19,
        "X": 20,
    }

    one_hot_seq = torch.zeros(len(seq), 21)
    for i, aa in enumerate(seq):
        one_hot_seq[i, aa_to_num[aa]] = 1

    return one_hot_seq


def return_loaders(
    data,
    bs,
    use_aa,
    use_sf,
    channels_first=True,
    max_len_aa=None,
    max_len_ss=None,
    val=True,
    device="cuda",
    drop_last=False,
):
    """
    Prepare the data for training and validation

    Parameters
    ----------
    data : dict
        Dictionary containing the data for training and validation
        keys: X_train_esm, X_val_esm, X_train_sf, X_val_sf, y_train, y_val
    returns:
        Dictionary containing the train and validation data loaders
    """
    assert use_aa or use_sf, "At least one of the features should be used"

    loaders = {}

    y_train = torch.tensor(data["y_train"]).unsqueeze(1).float().to(device)
    if val:
        y_val = torch.tensor(data["y_val"]).unsqueeze(1).float().to(device)

    i = 2 if channels_first else 1

    if use_aa:
        if max_len_aa is None:
            max_len_aa = max([x.size()[i] for x in data["X_train_aa"]])

        padded_X_train_aa = pad_features(
            data["X_train_aa"], max_len=max_len_aa, channels_first=channels_first
        ).to(device)

        if val:
            padded_X_val_aa = pad_features(
                data["X_val_aa"], max_len=max_len_aa, channels_first=channels_first
            ).to(device)

    if use_sf:
        if val:
            max_len_ss = max([x.size()[i] for x in data["X_train_sf"]])
        padded_X_train_ss = pad_features(
            data["X_train_sf"], max_len=max_len_ss, channels_first=channels_first
        ).to(device)

        if val:
            padded_X_val_ss = pad_features(
                data["X_val_sf"], max_len=max_len_ss, channels_first=channels_first
            ).to(device)

    if use_aa and use_sf:
        train_ds = AA_SS_Dataset(padded_X_train_aa, padded_X_train_ss, y_train)
        if val:
            val_ds = AA_SS_Dataset(padded_X_val_aa, padded_X_val_ss, y_val)

    elif use_sf and not use_aa:
        train_ds = AA_Dataset(padded_X_train_ss, y_train)
        if val:
            val_ds = AA_Dataset(padded_X_val_ss, y_val)

    elif use_aa and not use_sf:
        train_ds = AA_Dataset(padded_X_train_aa, y_train)
        if val:
            val_ds = AA_Dataset(padded_X_val_aa, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,# drop_last=drop_last
    ) 
    loaders["train_loader"] = train_loader

    if val:
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        loaders["val_loader"] = val_loader

    return loaders


def return_aa_ss_shapes(loaders, use_aa, use_sf):
    if use_sf and use_aa:
        aa_features_shape = loaders["train_loader"].dataset[0][0][0].size()
        ss_features_shape = loaders["train_loader"].dataset[0][0][1].size()
    elif use_aa and not use_sf:
        aa_features_shape = loaders["train_loader"].dataset[0][0].size()
        ss_features_shape = (0, 0)
    elif use_sf and not use_aa:
        aa_features_shape = (0, 0)
        ss_features_shape = loaders["train_loader"].dataset[0][0].size()

    return aa_features_shape, ss_features_shape


def custom_stratify(data, features=["labels", "CRISPR_system"], encode=True):
    """
    This function is used to stratify the data based on the Acr family, the inhibition types, and the labels
    """

    df = pd.DataFrame(columns=features)

    if "labels" in features:
        labels = [d["inhibition"] for d in data]
        df["labels"] = labels

    if "Acr_families" in features:
        Acr_families = [
            d["Acr_family"].split()[0].split(".")[0].split("-")[0] for d in data
        ]
        Acr_families = [re.sub(r"\d+", "", i) for i in Acr_families]  # remove numbers
        if encode:
            Acr_families = pd.factorize(Acr_families)[0]

        df["Acr_families"] = Acr_families

    if "CRISPR_system" in features:
        CRISPR_system = [d["CRISPR_system"] for d in data]
        if encode:
            CRISPR_system = pd.factorize(CRISPR_system)[0]
        df["CRISPR_system"] = CRISPR_system

    return df


def prediction_stats(preds, val_data, verbose=0, plot=False):
    """
    This function shows information about the predictions.
    It displays the number of correct and mistaken predictions for each CRISPR subsystem.

    Parameters
    ----------
    preds : list
        List of predictions by the model. The predictions are considered to be in one-hot format.
    val_data : list
        The data object for the predictions.

    Returns
    -------
    res: dict
        A dictionary of the results. Each subsystem is a key, and the value is a dict
        with the number of correct and wrong predictions for that subsystem, as well as accuracy and F1 score.
        Example: res = {"I-E": {"correct": 10, "wrong": 5, "accuracy": 0.667, "f1": 0.8},
    """
    # preds = np.array(preds).argmax(axis=1)
    labels = np.array([v["inhibition"] for v in val_data])
    sys_pred_pair = [(p, v["CRISPR_system_type"]) for p, v in zip(preds, val_data)]
    sys_lbl_pair = [(l, v["CRISPR_system_type"]) for l, v in zip(labels, val_data)]

    correct = preds == labels
    wrong = preds != labels

    CRISPR_systems = [
        v["CRISPR_system"] + "_" + v["CRISPR_system_type"] for v in val_data
    ]
    wrong_names = [CRISPR_systems[i] for i in range(len(wrong)) if wrong[i]]
    correct_names = [CRISPR_systems[i] for i in range(len(correct)) if correct[i]]

    unique_names = sorted(set(wrong_names) | set(correct_names))

    system_stat = {
        "I-F": {
            "correct": 0,
            "wrong": 0,
        },
        "I-E": {
            "correct": 0,
            "wrong": 0,
        },
        "I-C": {
            "correct": 0,
            "wrong": 0,
        },
    }

    for name in unique_names:
        subsystem = name.split("_")[-1]  # I-F, I-E, or I-C
        system_stat[subsystem]["correct"] += correct_names.count(name)
        system_stat[subsystem]["wrong"] += wrong_names.count(name)

    for subsystem in system_stat:
        system_stat[subsystem]["Accuracy"] = round(
            system_stat[subsystem]["correct"]
            / (system_stat[subsystem]["correct"] + system_stat[subsystem]["wrong"]),
            2,
        )
        labls_sys = [i[0] for i in sys_lbl_pair if i[1] == subsystem]
        preds_sys = [i[0] for i in sys_pred_pair if i[1] == subsystem]
        system_stat[subsystem]["F1"] = round(
            f1_score(labls_sys, preds_sys, average="weighted"), 2
        )

    if plot:  # show the F1 score for each subsystem
        plt.figure(figsize=(5, 5))
        plt.bar(system_stat.keys(), [system_stat[i]["F1"] for i in system_stat])
        plt.title("F1 score for each CRISPR subsystem")
        plt.show()

    return system_stat


def return_AUC(y_preds, y_val, plot_ROC=False, save_path=None, title=None):
    """
    This function returns the AUC and AUPR score and plots the ROC curve

    Parameters
    ----------
    y_preds : list
        List of predictions, if using 2 nodes, then list should be the probability of the positive class
    plot_ROC : bool, optional
        Whether to plot the ROC curve, by default True

    Returns
    -------
    dict
        Dictionary of AUC score and AUCPR score
    """

    fpr, tpr, _ = roc_curve(y_val, y_preds)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_val, y_preds)
    aucpr_score = auc(recall, precision)

    if plot_ROC or save_path is not None:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC: {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if title is not None:
            plt.title(title)
        plt.legend(loc="lower right")
        if save_path is not None:
            os.makedirs(save_path[: save_path.rfind("/")], exist_ok=True)
            plt.savefig(f"{save_path}/ROC.jpg", dpi=300, bbox_inches="tight")
        if plot_ROC:
            plt.show()
        else:
            plt.close()

    return {"AUC": auc_score, "AUPR": aucpr_score}


def plot_train_val_data(train_data, val_data=None, rot=70, save_path=None, plot=False):
    _, ax = plt.subplots(1, 4, figsize=(15, 3))
    y_train = [i["inhibition"] for i in train_data]

    sns.countplot(x=y_train, ax=ax[0])
    ax[0].set_title("Train Label Dist")
    train_data = sorted(train_data, key=lambda x: x["CRISPR_system"])
    sns.countplot(x=[i["CRISPR_system"] for i in train_data], ax=ax[2])
    ax[2].set_title("Train CRISPR Systems")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=rot)

    if val_data is not None:
        y_val = [i["inhibition"] for i in val_data]
        sns.countplot(x=y_val, ax=ax[1])
        ax[1].set_title("Test Label Dist")
        val_data = sorted(val_data, key=lambda x: x["CRISPR_system"])
        sns.countplot(x=[i["CRISPR_system"] for i in val_data], ax=ax[3])
        ax[3].set_title("Test CRISPR Systems")
        ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=rot)

    if save_path != None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if plot:
        plt.show()


def plot_train_test_lbl_sys(train_data, test_data, rot=70, save_path=None, plot=False):
    def preprocess_data(data):
        grouped = (
            data[data["use"] == 1]
            .groupby(["CRISPR_name_short", "inhibition"])
            .size()
            .reset_index(name="count")
        )
        # Replace the strain and system names as required
        grouped["CRISPR_name_short"] = grouped["CRISPR_name_short"].replace(
            {
                "K12": "K12_IE",
                "PA14": "PA14_IF",
                "PaLML1_DVT419": "PaLML1_IC",
                "SCRI1043": "SCRI1043_IF",
                "SMC4386": "SMC4386_IE",
            }
        )

        # Replace the heading of inhibition column with Inhibition
        grouped = grouped.rename(columns={"inhibition": "Inhibition"})

        return grouped

    train_grouped = preprocess_data(train_data)
    test_grouped = preprocess_data(test_data)

    train_pivoted = train_grouped.pivot(
        index="CRISPR_name_short", columns="Inhibition", values="count"
    ).fillna(0)

    test_pivoted = test_grouped.pivot(
        index="CRISPR_name_short", columns="Inhibition", values="count"
    ).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    train_pivoted.plot(kind="bar", stacked=True, rot=rot, ax=ax1)
    ax1.set_title("Train Dataset")
    ax1.set_xlabel("strain and system type")
    ax1.set_ylabel("Count")

    test_pivoted.plot(kind="bar", stacked=True, rot=rot, ax=ax2)
    ax2.set_title("Test Dataset")
    ax2.set_xlabel("strain and system type")
    ax2.set_ylabel("Count")

    plt.suptitle("Distribution of Training and Test Data by Strain and Label")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if plot:
        plt.show()


def split_data(
    inhibition_df,
    data,
    X_aa,
    X_ss,
    test_size=0.2,
    return_val=False,
    write_to_file=False,
):
    """
    returns: train_df, test_df, train_data, test_data, X_train_aa, X_test_aa, X_ss_train, X_ss_test
    """

    (
        train_df,
        test_df,
        train_data,
        test_data,
        X_train_aa,
        X_test_aa,
        X_train_sf,
        X_test_sf,
    ) = train_test_split(
        inhibition_df,
        data,
        X_aa if X_aa is not None else [None for _ in range(len(data))],
        X_ss if X_ss is not None else [None for _ in range(len(data))],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=custom_stratify(data),
    )

    if return_val:
        (
            val_df,
            test_df,
            val_data,
            test_data,
            X_val_aa,
            X_test_aa,
            X_val_sf,
            X_test_sf,
        ) = train_test_split(
            test_df,
            test_data,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=custom_stratify(test_data),
        )

    if write_to_file:
        train_df.to_csv(f"{VERSION_FOLDER}/train_val.csv", index=False)
        val_df.to_csv(f"{VERSION_FOLDER}/val.csv", index=False)
        test_df.to_csv(f"{VERSION_FOLDER}/test.csv", index=False)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_data": train_data,
        "test_data": test_data,
        "val_df": val_df if return_val else None,
        "val_data": val_data if return_val else None,
        "X_train_aa": X_train_aa,
        "X_test_aa": X_test_aa,
        "X_aa_val": X_val_aa if return_val else None,
        "X_train_sf": X_train_sf,
        "X_test_sf": X_test_sf,
        "X_val_sf": X_val_sf if return_val else None,
    }


def undersample_PaLML1_SMC4386(inhibition_df):
    """Drop 19 negative samples from PaLML1_IC and 18 negative samples from SMC4386 to balance the dataset"""
    df_to_drop1 = (
        inhibition_df[
            (inhibition_df["CRISPR_name_short"] == "PaLML1_DVT419")
            & (inhibition_df["inhibition"] == 0)
        ]
        .sample(n=19, random_state=RANDOM_STATE)
        .index
    )
    df_to_drop2 = (
        inhibition_df[
            (inhibition_df["CRISPR_name_short"] == "SMC4386")
            & (inhibition_df["inhibition"] == 0)
        ]
        .sample(n=18, random_state=RANDOM_STATE)
        .index
    )

    inhibition_df = inhibition_df.drop(df_to_drop1)
    inhibition_df = inhibition_df.drop(df_to_drop2)

    return inhibition_df


def group_system_label(inhibition_df):
    grouped = (
        inhibition_df[inhibition_df["use"] == 1]
        .groupby(["CRISPR_name_short", "inhibition"])
        .size()
        .reset_index(name="count")
    )
    # replace the K12 with K12_IE, PA14 with PA14_IF, PaLML1_DVT419 with PaLML1_IC, SCRI1043 with SCRI1043_IF
    # SMC4386 with SMC4386_IE in the CRISPR_name_short column
    grouped["CRISPR_name_short"] = grouped["CRISPR_name_short"].replace(
        {
            "K12": "K12_IE",
            "PA14": "PA14_IF",
            "PaLML1_DVT419": "PaLML1_IC",
            "SCRI1043": "SCRI1043_IF",
            "SMC4386": "SMC4386_IE",
        }
    )

    # replace the heading of inhibition column with Inhibition
    grouped = grouped.rename(columns={"inhibition": "Inhibition"})
    return grouped


def extract_combined_features(data, config, verbose=0):
    """
    This function calculates the hidden states (ESM and SS) for the Acr and Cas sequences
    and then concatenates the Acr and Cas hidden states

    Parameters:
    ----------------
    data (list):
        list of dictionaries containing the Acr and Cas sequences
    config (dict):
        dictionary containing the configuration parameters.
        config["feature_mode"] = 0: just ESM
        config["feature_mode"] = 1: one-hot encoding of amino acids
        config["feature_mode"] = 3: for just SS, no ESM

        config["Acr_Cas_mode"] = 1: just Cas
        config["Acr_Cas_mode"] = 2: Acr and Cas

        config["channels_first"] = True: the output tensor will have the shape (batch_size, channels, seq_len)
        config["channels_first"] = False: the output tensor will have the shape (batch_size, seq_len, channels)

    verbose (int):
        verbose level

    Returns:
    ----------------
    features (list):
        list of tensors
    """

    if verbose:
        print("Extracting features ...")
        print(f"Feature mode: {config['feature_mode']}")
        print("channels_first: ", config["channels_first"])

    features = []

    for i, dict in enumerate(data):
        states = []
        crispr_states = []
        cas_proteins = dict["Cas_proteins"]
        cas_ids = dict["Cas_ids"]

        ########################### Acr ###########################
        if config["Acr_Cas_mode"] > 1:
            acr = dict["Acr_seq"]
            acr_id = dict["Acr_id"]
            acr_state = None

            if config["feature_mode"] == 0:  # ESM
                acr_state = extract_hidden_state(acr)
            elif config["feature_mode"] == 3:  # SS
                acr_ss = return_ss_pt(acr_id, ss_df_path=config["Acr_ss_df"])
                acr_state = acr_ss.unsqueeze(0)
            elif config["feature_mode"] == 1:  # one-hot encoding AA
                acr_state = one_hot_enc(acr)
                acr_state = acr_state.unsqueeze(0)

            if config["channels_first"]:
                acr_state = acr_state.permute(0, 2, 1)

            states.append(acr_state)

        ########################### CRISPR ###########################
        # ESM
        if config["feature_mode"] == 0:
            crispr_states = extract_hidden_state(cas_proteins)
            if config["channels_first"]:
                crispr_states = [state.permute(0, 2, 1) for state in crispr_states]

        # one-hot encoding
        elif config["feature_mode"] == 1:
            for cas in cas_proteins:
                crispr_state = one_hot_enc(cas)
                crispr_state = crispr_state.unsqueeze(0)
                if config["channels_first"]:
                    crispr_state = crispr_state.permute(0, 2, 1)

                crispr_states.append(crispr_state)

        # SF
        elif config["feature_mode"] == 3:
            for j in range(len(cas_ids)):
                # get the secondary structure features for the Cas proteins
                crispr_ss = return_ss_pt(cas_ids[j], ss_df_path=config["CRISPR_ss_df"])

                crispr_ss = crispr_ss.unsqueeze(0)
                if config["channels_first"]:
                    crispr_ss = crispr_ss.permute(0, 2, 1)
                    # print(crispr_ss.shape)

                crispr_states.append(crispr_ss)

        if config["CRISPR_mode"] == "concat":
            states.extend(crispr_states)
        # elif config["CRISPR_mode"] == "sum":
        #     states.append(sum_tensors(crispr_states))

        states_concat = torch.cat(states, dim=2 if config["channels_first"] else 1)

        if config["Acr_Cas_mode"] == 1:  # Cas only mode
            features.append(
                (dict["CRISPR_system"], states_concat)
            )  # if just crispr, add the system name to the features

        else:
            features.append(states_concat)

        if verbose:
            print(f"features extracted {i+1}/{len(data)}! size: {states_concat.shape}")

    return features


def extract_hidden_state(seqs):
    """
    This function extracts the hidden states for a list of sequences

    Parameters
    ----------
    seqs : list
        list of sequences

    Returns
    -------
    list
        list of hidden states
    """

    if type(seqs) == str:
        return comp_hidden_states(seqs)[-1]

    states = []
    for seq in seqs:
        cas_state = comp_hidden_states(seq)[-1]
        states.append(cas_state)

    return states


def load_extrated_features(feature_mode, features_config, data):
    feature_dir = f"{VERSION_FOLDER}/pkl/{PROJ_VERSION}/"
    features_file_name = f"_{features_config['features_names']}_rs{RANDOM_STATE}.pkl"

    features_aa, features_sf = None, None

    exclude_mode_dict = features_config["exclude_mode_dict"]
    excl_mode = features_config["exclude_mode"]
    if exclude_mode_dict[excl_mode] is not None:
        features_file_name = features_file_name.replace(
            ".pkl", f"_excl_{'_'.join(exclude_mode_dict[excl_mode])}.pkl"
        )
    # if undersample:
    #     features_file_name = features_file_name.replace(".pkl", "_undersample.pkl")

    # AA features
    if feature_mode == 3:
        features_config[
            "feature_mode"
        ] = 1  # 0 for just ESM | 1 for one-hot encoding of AA| 3 for just sf |
        features_aa = extract_combined_features(data, features_config, verbose=1)

    # Structural Features
    if feature_mode == 2 or feature_mode == 4:
        features_config["feature_mode"] = 3
        os.makedirs(feature_dir, exist_ok=True)
        ffn_sf = "SF" + features_file_name
        if os.path.exists(feature_dir + ffn_sf) != True:
            features_sf = extract_combined_features(data, features_config, verbose=1)
            with open(feature_dir + ffn_sf, "wb") as f:
                pickle.dump(features_sf, f)
        else:
            features_sf = pickle.load(open(feature_dir + ffn_sf, "rb"))

    # ESM features
    if feature_mode == 1 or feature_mode == 4:
        features_config["feature_mode"] = 0
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

    return features_aa, features_sf


def prepare_loaders(
    split_dict,
    use_aa,
    use_sf,
    device,
):
    """
    This function prepares the data loaders for training and validation
    and returns: loaders, seq_len_aa, seq_len_sf
    """
    
    if use_aa:
        X_train_aa = split_dict["X_train_aa"]
        X_test_aa = split_dict["X_test_aa"]
    if use_sf:
        X_train_sf = split_dict["X_train_sf"]
        X_test_sf = split_dict["X_test_sf"]

    train_data = split_dict["train_data"]
    test_data = split_dict["test_data"]

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

    loaders = return_loaders(
        loader_input,
        use_aa=use_aa,
        use_sf=use_sf,
        bs=BS,
        channels_first=True,
        val=True,
        device=device,
    )

    if use_sf and use_aa:
        seq_len_aa = loaders["train_loader"].dataset[0][0][0].size()[1]
        seq_len_sf = loaders["train_loader"].dataset[0][0][1].size()[1]
    elif use_aa and not use_sf:
        seq_len_aa = loaders["train_loader"].dataset[0][0].size()[1]
        seq_len_sf = 0
    elif use_sf and not use_aa:
        seq_len_aa = 0
        seq_len_sf = loaders["train_loader"].dataset[0][0].size()[1]

    return loaders, seq_len_aa, seq_len_sf


def compute_class_weights(y_train, device):
    class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
    class_weights = torch.tensor(
        class_weights,
        device=device,
        dtype=torch.float,
    )
    return class_weights