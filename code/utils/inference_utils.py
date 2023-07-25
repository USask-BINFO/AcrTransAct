import os
import torch
import sys
sys.path.append('code/')
from utils.data_utils import *
from utils.model_utils import *
from utils.ppi_utils import *
from utils.ss_utils import *
from datetime import datetime
from utils.model_utils import *

def write_test_results(results, results_dir, results_file_name):
    with open(f"{results_dir}/{results_file_name}.txt", "w") as f:
        for dic in results:
            for seq in dic.keys():
                f.write(f"======================== {seq} ========================\n")
                for system in dic[seq].keys():
                    f.write(
                        f"{'_'.join(system.split('_')[1:])}\t, probability of inhibition: {dic[seq][system]:.3f}\n"
                    )
    print(f"wrote the results to {results_dir}/{results_file_name}.txt")

def Acr_existing_CRISPR_features(
    seq_dics, folder_cr_features, max_len, use_ss, del_netsurf_res=False
):
    """
    This function extracts the features of the Acr sequences and concat it
    with all of the existing CRISPR features.

    Parameters
    ----------
    seq_dics : dict
        dictionary of the headers and sequences of the Acr sequences
    folder_cr_features : str
        folder address of the CRISPR features. Can be the ESM or SS feature folder.
    max_len : int
        maximum length of the sequences

    Returns
    -------
    results : dict
        dictionary that its keys are the headers of the Acr sequences and its values
        are the features of different CRISPR systems and the Acr sequence of that header
        concatenated together.

        {header_Acr_seq_1: {cr_sys1: features_cr_sys1, cr_sys2: features_cr_sys2, ...}
         header_Acr_seq_2: {cr_sys1: features_cr_sys1, cr_sys2: features_cr_sys2, ...}
        }
    """
    results = {}
    Acr_results = []

    # save the sequences.
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    Acrs_biolib_path = f"biolib_results_inference/Acrs/{time_stamp}"
    os.makedirs(Acrs_biolib_path, exist_ok=True)
    Acrs_fasta_path = f"{Acrs_biolib_path}/Acrs.fasta"
    for header in seq_dics.keys():
        with open(Acrs_fasta_path, "a") as f:
            f.write(f">{header}\n{seq_dics[header]}\n")

    if use_ss:  # SS only
        Acr_results = retrieve_ss_for_test(Acrs_fasta_path)

        # if del_netsurf_res, delete files and folder under Acrs_biolib_path
        if del_netsurf_res:
            os.rmdir(Acrs_biolib_path)

    else:  # ESM only
        with torch.no_grad():
            load_feature_extractor(features_config)
            for header in seq_dics.keys():
                state = extract_hidden_state(seq_dics[header])
                Acr_results.append((header, state.float().cpu()))

    ############################################################################################################
    CR_pkl_files = os.listdir(folder_cr_features)
    CR_pkl_files = [file for file in CR_pkl_files if file.endswith(".pkl")]

    # attach the input Acrs to pre-computed CRISPR features
    for header, acr_state in Acr_results:
        acr_state = acr_state.permute(0, 2, 1) #permutate for channel first
        features_systems_dic = {}
        for cr_pkl in CR_pkl_files:
            # get the systm name from the file name
            system_name = cr_pkl.split("_")[-3:]
            system_name = (
                "_".join(system_name[-3:])
                .split(".")[0]
                .replace("ESM8m_", "")
                .replace("_ss", "")
                .replace("_esm", "")
            )

            cr_features = pickle.load(open(f"{folder_cr_features}/{cr_pkl}", "rb"))
            cr_features = torch.tensor(cr_features).float().cpu()

            # print("* acr_state.shape:", acr_state.shape)
            # print("* cr_features.shape:", cr_features.shape)

            concat_features = torch.cat([acr_state, cr_features], axis=2)
            padd_features = pad_features([concat_features], max_len)
            features_systems_dic[system_name] = padd_features

        header = header.replace(
            "/", ""
        )  # remove the / from the header because biolib does this, I have to do it so it matches the headers of ESm
        results[header] = features_systems_dic

    return results

def predict_against_existing(
    model_cls,
    input_fasta_addr,
    max_len_esm,
    max_len_ss=None,
    use_ss=False,
    folder_crispr_esm_features="inference/data/pkl/CRISPR/esm/",
    folder_crispr_sf_features="inference/data/pkl/CRISPR/sf/"
):
    """
    This function takes an Acr input and predicts against all the CRISPR systems
    in the data/pkl/{data_version}/CRISPR folder

    Parameters
    ----------
    Acr_input : str
        The Acr input sequence
    use_ss : bool, optional
        If True, the model will use the SF features, by default False

    Returns
    -------
    results : list
        A list of tuples containing the system name and the prediction score
    """

    results = []

    Acr_dics = read_fasta(input_fasta_addr)

    # ESM features
    features_esm = Acr_existing_CRISPR_features(
        Acr_dics, folder_crispr_esm_features, max_len_esm, use_ss=False
    )

    # SF features
    if use_ss:
        features_ss = Acr_existing_CRISPR_features(
            Acr_dics, folder_crispr_sf_features, max_len_ss, use_ss=True
        )
        # make sure the systems are the same in both ESM and SS features
        assert sorted(features_esm.keys()) == sorted(features_ss.keys()), print(
            "The headers in ESM and SF features don't match"
        )

    for seq in features_esm.keys():
        pred_acr_dic = {}  # a dictionary to store the prediction for each system
        for system in features_esm[seq].keys():
            esm_fe = features_esm[seq][system]
            esm_fe = esm_fe

            if use_ss:
                ss_fe = features_ss[seq][system]
                ss_fe = ss_fe
                fe_in = (esm_fe, ss_fe)
            else:
                fe_in = esm_fe

            pred = model_cls(fe_in).cpu().detach().numpy()[0][1]
            pred_acr_dic[system] = pred

        results.append({seq: pred_acr_dic})

    print("Prediction is done!")

    return results
