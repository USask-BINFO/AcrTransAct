import re
import torch
from Bio import SeqIO
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
from utils.common_vars import *
from sklearn.model_selection import train_test_split
from collections import Counter


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


def find_nums(input_str):
    """This function finds the numbers in a string and returns them as a list
    ----------
    Returns
    -------
    nums : list
        A list of string which are numbers found in the input"""

    nums = re.findall(r"\d+", input_str)
    return nums


def TSNE_torch(input_tensor, n_components=50):
    """
    Applies t-SNE on a PyTorch tensor.
    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor.
    Returns
    -------
    tensor_tsne_torch : torch.Tensor
        t-SNE result.
    """

    tensor_np = input_tensor.cpu().numpy()

    tsne = TSNE(n_components=n_components)
    tensor_tsne = tsne.fit_transform(tensor_np)

    tensor_tsne_torch = torch.from_numpy(tensor_tsne)

    return tensor_tsne_torch


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
    # Amino acid to number dictionary
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
    device="cuda"
):
    """
    Prepare the data for training and validation

    Parameters
    ----------
    data : dict
        Dictionary containing the data for training and validation
        keys: X_train_esm, X_val_esm, X_train_ss, X_val_ss, y_train, y_val
    bs : int, optional
        Batch size, by default 64
    use_aa : bool,
        Whether to create a dataset with AA features, by default True
    use_sf : bool,
        Whether to create a dataset with SS features, by default True
    channels_first : bool, optional
        Whether to use channels first or last, by default True
    max_len_aa : int, optional
        Maximum length of the AA sequences, by default None
    max_len_ss : int, optional
        Maximum length of the SS sequences, by default None
    val: bool, optional
        Whether to create a validation set, by default True
    device: str, optional
        Device to move the tensors (cuda or cpu), by default "cuda"

    Returns
    -------
    loaders : dict
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
        train_ds, batch_size=bs, shuffle=True
    )  # shuffle the training data
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


from collections import Counter
from sklearn.metrics import f1_score

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
    preds = np.array(preds).argmax(axis=1)
    labels = np.array([v["inhibition"] for v in val_data])
    sys_pred_pair = [(p, v["CRISPR_system_type"]) for p, v in zip(preds, val_data)]
    sys_lbl_pair = [(l, v["CRISPR_system_type"]) for l, v in zip(labels, val_data)]

    correct = preds == labels
    wrong = preds != labels

    CRISPR_systems = [v["CRISPR_system"]+"_"+v["CRISPR_system_type"] for v in val_data]
    wrong_names = [CRISPR_systems[i] for i in range(len(wrong)) if wrong[i]]
    correct_names = [CRISPR_systems[i] for i in range(len(correct)) if correct[i]]

    unique_names = sorted(set(wrong_names) | set(correct_names))

    system_stat = {
        "I-F": {"correct": 0, "wrong": 0, },
        "I-E": {"correct": 0, "wrong": 0, },
        "I-C": {"correct": 0, "wrong": 0, }
    }

    for name in unique_names:
        subsystem = name.split("_")[-1] # I-F, I-E, or I-C
        system_stat[subsystem]["correct"] += correct_names.count(name)
        system_stat[subsystem]["wrong"] += wrong_names.count(name)

    for subsystem in system_stat:
        system_stat[subsystem]["Accuracy"] = round(system_stat[subsystem]["correct"] / (
                system_stat[subsystem]["correct"] + system_stat[subsystem]["wrong"]
        ), 2)
        labls_sys = [i[0] for i in sys_lbl_pair if i[1] == subsystem]
        preds_sys = [i[0] for i in sys_pred_pair if i[1] == subsystem]
        system_stat[subsystem]["F1"] =  round(f1_score(labls_sys, preds_sys, average="weighted"), 2)

    if plot: # show the F1 score for each subsystem
        plt.figure(figsize=(5, 5))
        plt.bar(system_stat.keys(), [system_stat[i]["F1"] for i in system_stat])
        plt.title("F1 score for each CRISPR subsystem")
        plt.show()

    return system_stat


def return_AUC(y_preds, y_val, plot_ROC=False, save_path=None):
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
        plt.title(f"ROC Curve")
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
        ax[1].set_title("Validation Label Dist")
        val_data = sorted(val_data, key=lambda x: x["CRISPR_system"])
        sns.countplot(x=[i["CRISPR_system"] for i in val_data], ax=ax[3])
        ax[3].set_title("Validation CRISPR Systems")
        ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=rot)

    if save_path != None:
        plt.savefig(f"{VERSION_FOLDER}/label_dist.jpg", dpi=300, bbox_inches="tight")

    if plot:
        plt.show()


def split_data(
    inhibition_df,
    data,
    X_aa,
    X_ss,
    test_size=0.15,
    return_val=False,
    write_to_file=False,
):
    """
    returns: train_df, test_df, train_data, test_data, X_train_aa, X_test_aa, X_ss_train, X_ss_test
    """

    (
        train_df, test_df, train_data, test_data, X_train_aa, X_test_aa, X_train_sf, X_test_sf
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
            val_df, test_df, val_data, test_data, X_val_aa, X_test_aa, X_val_sf, X_test_sf 
        )= train_test_split(
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
    df_to_drop1 = inhibition_df[(inhibition_df["CRISPR_name_short"] == "PaLML1_DVT419") & (inhibition_df["inhibition"] == 0)]\
                    .sample(n=19, random_state=RANDOM_STATE).index
    df_to_drop2= inhibition_df[(inhibition_df["CRISPR_name_short"] == "SMC4386") & (inhibition_df["inhibition"] == 0)]\
                    .sample(n=18, random_state=RANDOM_STATE).index
    
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