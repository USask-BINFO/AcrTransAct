import pandas as pd
import os
import biolib
import numpy as np
import torch
import json
from utils.data_utils import find_nums
from sklearn.preprocessing import OneHotEncoder


def str_list_to_float_list(str_list):
    """
    This function converts a string list to a float list

    Parameters:
    ----------------
    str_list (str):
        string list

    Returns:
    ----------------
    int_list (list):
        float list

    Example:
    ----------------
    str_list = "[1, 2, 3]"
    int_list = str_list_to_float_list(str_list)
    print(int_list)
    >>> [1.0, 2.0, 3.0]
    """

    str_list = str_list.replace("[", "")
    str_list = str_list.replace("]", "")
    str_list = str_list.split(",")
    int_list = [float(i) for i in str_list]
    return int_list


def read_2nd_csv(path, q=3):
    """
    This function reads the csv file with the secondary structure information

    --------------------------------
    Parameters:

    path : str
        The path to the NetSurfP-3 csv file with the secondary structure information
    q : int
        The q value to use (3 or 8)
    --------------------------------
    Returns:

    sec_struct : list
        A list of letters (H, S, T) that represent where each token is located in the secondary structure
    """

    data = pd.read_csv(path)

    sec_struct = data[f" q{q}"].tolist()

    return sec_struct


def check_2nd_area(tokens, seq_name, path, q=3):
    """
    This function checks if the input token is inside the region
    that codes for the secondary structure area (helices or sheets)
    --------------------------------
    Parameters:

    token: list
        the list of tokens to check
    sec_struct: list
        a list of letters (H, S, T) that represent where each token
        is located in the secondary structure
    q: int
        the q value to use (3 or 8)
    --------------------------------
    Returns:

    c : int
        the number of tokens that are inside the secondary structure area
    """

    for p in os.listdir(path):
        if p.endswith(".csv") and p.split("_")[1][:-4] == seq_name:
            path = os.path.join(path, p)
            break

    sec_struct = read_2nd_csv(path=path, q=q)
    c = 0

    for token in tokens:
        nums = find_nums(token)
        i = 0
        if len(nums) > 1:
            i = 1

        # H = helix, E = sheet
        if sec_struct[int(nums[i]) - 1] == "H" or sec_struct[int(nums[i]) - 1] == "E":
            c += 1

    return c


def run_netsurf(input_fasta_path, output_folder):
    """
    This function uses the NetSurfP-3.0 to extract the secondary structure
    of the sequences in the input_path

    Parameters
    ----------
    input_path:
        path to the fasta file of the Acrs
    output_folder:
        path to the folder where the results will be saved

    Returns
    -------
    None
    """

    # check if the input path exists
    if not os.path.exists(input_fasta_path):
        raise ValueError("The input path does not exist.")

    nsp3 = biolib.load("DTU/NetSurfP-3")
    nsp3_results = nsp3.cli(args=f"-i {input_fasta_path}")
    nsp3_results.save_files(output_folder)
    print(
        f"NetSurfP-3.0 analysis finished. Results saved in the {output_folder} folder."
    )


def create_df_ss(results_dir):
    """
    Create a dataframe of the structural features from the results in the results_dir folder.

    Parameters
    ----------
    results_dir : str
        The path to the folder containing the results of the netsurfp analysis.

    Returns
    -------
    netsurfp_df : pd.DataFrame
        A dataframe containing the secondary structures of the sequences.
    """

    folders = [
        f for f in os.listdir(results_dir) if os.path.isdir(f"{results_dir}/{f}")
    ]

    # create a dataframe to save the results
    netsurfp_df = pd.DataFrame()

    dfs = []

    for folder in folders:
        with open(f"{results_dir}/{folder}/{folder}.json") as f:
            data = json.load(f)

        df_item = {
            "id": data["desc"],
            "seq": data["seq"],
            "q3": data["q3"],
            "q8": data["q8"],
            "asa": data["asa"],
            "rsa": data["rsa"],
            "disorder": data["disorder"],
        }
        # create a new row in the dataframe
        dfs.append(df_item)

    netsurfp_df = pd.DataFrame(dfs, index=None)
    # sort the dataframe by the id column
    netsurfp_df = netsurfp_df.sort_values(by="id")
    # reset the index
    netsurfp_df = netsurfp_df.reset_index(drop=True)

    return netsurfp_df


def return_ss_pt(id, ss_df_path=None, ss_df=None, features=["q3", "asa", "rsa", "disorder"]):
    """
    This function returns the structural features for a given protein id
    as pytorch tensor

    Parameters:
    ----------------
    id (str):
        protein id
    ss_df_path (str):
        path to the secondary structure dataframe
    ss_df (pd.DataFrame):
        secondary structure dataframe passed as argument. If this argument is
        passed, the ss_df_path argument is ignored
    features (list):
        list of features to return. The options are: q3, q8, asa, rsa, disorder

    Returns:
    ----------------
    ss_pt (torch.tensor):
        tensor of shape (1, 6)
    """
    rff = False  # read from file
    feature_list = []

     # if the dataframe is read from excel file then convert the str lists ot python lists
    if ss_df is None:
        rff = True
        ss_df = pd.read_excel(ss_df_path)

    if "q3" in features:
        q3_state = ss_df[ss_df["id"] == id]["q3"].values[0]
        q3_state = torch.tensor(one_hot_encode_ss(q3_state))
        feature_list.append(q3_state)

    if "q8" in features:
        q8_state = ss_df[ss_df["id"] == id]["q8"].values[0]
        q8_state = torch.tensor(one_hot_encode_ss(q8_state))
        feature_list.append(q8_state)

    if "asa" in features:
        # get the asa, rsa and disorder from str format to list
        asa = ss_df[ss_df["id"] == id]["asa"].values[0]
        if rff: 
            asa = str_list_to_float_list(asa)
        asa = torch.tensor(asa).unsqueeze(1)
        feature_list.append(asa)
    
    if "rsa" in features:
        rsa = ss_df[ss_df["id"] == id]["rsa"].values[0]
        if rff:
            rsa = str_list_to_float_list(rsa)
        rsa = torch.tensor(rsa).unsqueeze(1)
        feature_list.append(rsa)
    
    if "disorder" in features:
        disorder = ss_df[ss_df["id"] == id]["disorder"].values[0]
        if rff:
            disorder = str_list_to_float_list(disorder)
        disorder = torch.tensor(disorder).unsqueeze(1)
        feature_list.append(disorder)

    ss_pt = torch.cat(feature_list, axis=1)

    return ss_pt


# TODO: replace the dictionary with SKLEAN OneHotEncoder
def one_hot_encode_ss(sec_struct, q=3):
    """
    This function one-hot encodes a secondary structure sequence.

    Parameters
    ----------
    sec_struct : str
        The secondary structure sequence to encode q3 or q8. e.g. "CCHHHHH"
    q : int
        The number of classes to encode. Either 3 or 8.

    Returns
    -------
    encoded_seq : np.array
        A numpy array of shape (len(sec_struct), 3 or 8) containing the one-hot encoding of the sequence.

    """

    # seq_chars = [i for i in sec_struct]

    # encoder = OneHotEncoder(sparse=False, max_categories= q, min_frequency=0)
    # encoded_seq = encoder.fit_transform(np.array(seq_chars).reshape(-1, 1))
    # return encoded_seq

    # Define the one-hot encoding dictionary
    q3_dict = {"C": [1, 0, 0], "E": [0, 1, 0], "H": [0, 0, 1]}

    # Initialize an empty list to store the encoded sequence
    encoded_seq = []

    # Iterate over each character in the secondary structure sequence
    for char in sec_struct:
        # If the character is not recognized, raise an error
        if char not in q3_dict:
            raise ValueError(
                f"Unrecognized character in secondary structure sequence: {char}"
            )
        # Otherwise, append the one-hot encoding to the list
        else:
            encoded_seq.append(q3_dict[char])

    # Convert the list to a numpy array and return it
    return np.array(encoded_seq)


def retrieve_ss_for_test(fasta_path, df_out_path=None):
    """
    This function creates a dataframe containing the secondary structure features for each sequence in the fasta file.

    Parameters
    ----------
    fasta_path : str
        The path to the fasta file containing the sequences.
    df_out_path : str
        The path to the output dataframe.

    Returns
    -------
    ss_features (list):
        list of tensors of ss features for each sequence in the fasta file. in the form of (seq_header, ss_features)
    """
    ss_features = []

    netsurf_res = f'{fasta_path.split(".fasta")[0]}'
    os.makedirs(netsurf_res, exist_ok=True)

    # TODO: delete
    # seqs = read_fasta(fasta_path, True)
    # for seq in seqs.keys():
    #     ss_features.append((seq, torch.zeros((1, 2350, 6))))

    # run netsurfp and save the results in the biolob_res folder
    # run_netsurf(fasta_path, netsurf_res)

    print(netsurf_res, " netsurf_res")
    netsurf_res = "biolib_results/Acrs/2023-04-22_18:30:29/Acrs"

    # create a dataframe by processing results in the biolob_res folder
    ss_df = create_df_ss(netsurf_res)

    # save the dataframe to an excel file
    if df_out_path is not None:
        ss_df.to_excel(df_out_path, index=False)

    for _, row in ss_df.iterrows():
        ss_features.append(
            (row["id"], return_ss_pt(row["id"], ss_df=ss_df).unsqueeze(0))
        )

    return ss_features
