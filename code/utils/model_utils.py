import torch
import wandb
import os
import json
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
from utils.common_vars import MONITOR_LR


from transformers import (
    EsmTokenizer,
    EsmModel,
    EsmForMaskedLM,
    EsmForSequenceClassification,
)

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
model, device, tokenizer = None, None, None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_features(feature_mode, esm_version):
    if feature_mode == 1:
        use_sf = False
        features_names = f"ESM{esm_version}"
        use_aa = True
    elif feature_mode == 2:
        use_sf = True
        features_names = "Just SF"
        use_aa = False
    elif feature_mode == 3:
        use_sf = False
        features_names = "One-hot AA"
        use_aa = True
    elif feature_mode == 4: # ESM + SF
        use_sf = True
        use_aa = True
        features_names = f"ESM{esm_version} + SF"

    return use_sf, use_aa, features_names



def load_feature_extractor(model_config, return_model=False):
    """
    Loads the model and the tokenizer

    Parameters
    ----------
    model_config : dict
        dictionary containing the model configuration
        which includes the model name, weights directory,
        task, output attentions and output hidden states.
        Available models: ESM3b, ESM650m, ESM150m, ESM35m, ESM8m, XL

    return_model : bool, optional
        if True, the model, tokenizer and device are returned, by default False

    Returns
    -------
    model, tokenizer, device
        if return_model is True, the model, tokenizer and device are returned
    """

    global model, device, tokenizer

    if model_config["device"] is not None:
        device = model_config["device"]
        print(f"Running the model on {device}")
    else:    
        if torch.cuda.is_available():
            # for CUDA
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            print("Running the model on CUDA")

        elif torch.backends.mps.is_available():
            # for M1
            device = torch.device("mps")
            print("Running the model on M1 CPU")

        else:
            print("Running the model on CPU")

    print(
        f"Loading model {model_config['model_name']}"
    )

    if model_config["model_name"].find("ESM") != -1:

        ESM_model_dic ={
        "ESM3b": "facebook/esm2_t36_3B_UR50D",
        "ESM650m": "facebook/esm2_t33_650M_UR50D",
        "ESM150m": "facebook/esm2_t30_150M_UR50D",
        "ESM35m": "facebook/esm2_t12_35M_UR50D",
        "ESM8m": "facebook/esm2_t6_8M_UR50D",
        }

        esm_name = model_config["model_name"].split("+")[0].strip()
        tokenizer = EsmTokenizer.from_pretrained(
            ESM_model_dic[esm_name], do_lower_case=False
        )

        if model_config["task"] == "analysis"\
              or model_config["task"] == "PPI":

            model = EsmModel.from_pretrained(
                ESM_model_dic[esm_name],
                output_attentions=model_config["output_attentions"],
                output_hidden_states=model_config["output_hidden_states"],
            )

        elif model_config["task"] == "MaskedLM":

            model = EsmForMaskedLM.from_pretrained(
                ESM_model_dic[model_config["model_name"]],
                output_attentions=model_config["output_attentions"],
            )

        elif model_config["task"] == "classification":

            model = EsmForSequenceClassification.from_pretrained(
                ESM_model_dic[model_config["model_name"]],
                output_attentions=model_config["output_attentions"],
                num_labels=model_config["num_labels"],
            )
        else:
            print("Task not supported")
            return

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["weights_dir"], do_lower_case=False
        )

        if model_config["task"] == "analysis" or model_config["task"] == "PPI":

            model = AutoModel.from_pretrained(
                model_config["weights_dir"],
                output_attentions=model_config["output_attentions"],
                output_hidden_states=model_config["output_hidden_states"],
            )

        elif model_config["task"] == "classification":

            model = AutoModelForSequenceClassification.from_pretrained(
                model_config["weights_dir"],
                output_attentions=model_config["output_attentions"],
                num_labels=model_config["num_labels"],
            )
        else:
            print("Task not supported")
            return

    model.eval()
    model.to(device)

    print(f'model {model_config["model_name"]} loaded')

    if return_model:
        return model, tokenizer, device


def comp_hidden_states(seq, verbose=0):
    """
    calculating hidden states for an input sequence
    
    Parameters
    ----------
    seq : str
        a sequence of amino acids
    verbose : int, optional
        if 1, the hidden states size is printed, by default 0
    
    Returns
    -------
    list
        list of hidden states for all layers
    """

    assert (
        model is not None
    ), "Load the model first"  # make sure the model is loaded first

    inputs = tokenizer.encode_plus(
        seq, return_tensors="pt", add_special_tokens=True, truncation = True
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        hidden_states = model(input_ids.to(device))["hidden_states"]

    if verbose > 0:
        print("hidden states size:", len(hidden_states), "x", hidden_states[0].shape)

    return hidden_states


def comp_self_attn(spaced_seq, verbose=0):
    """calculating self attention for input sequence
    the spaced_seq must be a sequence with space between
    amino acids
    """
    assert (
        model is not None
    ), "Load the model first"  # make sure the model is loaded first

    inputs = tokenizer.encode_plus(
        spaced_seq, return_tensors="pt", add_special_tokens=True, truncation = True
    )
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        attn = model(input_ids.to(device))["attentions"]

    if verbose > 0:
        print("attention size:", len(attn), "x", attn[0].shape)

    return attn


def comp_seq_to_seq_attn(spaced_seq1, spaced_seq2, verbose=0):
    """calculating self attention for input sequence
    spaced_seq1 and spaced_seq2 must be seqeunces with space
    between amino acids"""

    assert model != None, "Load the model first"  # make sure the model is loaded first

    inputs = tokenizer.encode_plus(
        spaced_seq1,
        spaced_seq2,
        return_tensors="pt",
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation = True
    )  # encode_plus encodes 2 sequences
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        attn = model(input_ids.to(device))["attentions"]

    token_type_ids = inputs[
        "token_type_ids"
    ]  # token_type_ids indicate which tokens are for 1st seq and which are for 2nd

    if len(token_type_ids[0]) == (
        len(spaced_seq1.split()) + len(spaced_seq2.split())
    ):  # special tokens not added by the tokenizer (ESM)
        sentence_b_start = len(spaced_seq1.split()) + 2
        print("special tokens not added, make sure your model is the ESM")

    elif len(token_type_ids[0]) == (
        len(spaced_seq1.split()) + len(spaced_seq2.split()) + 3
    ):  # special tokens added by the tokenizer (XLNET)
        sentence_b_start = token_type_ids[0].tolist().index(1)
        print("special tokens added model should not be ESM")

    # elif len(token_type_ids[0]) == (len(spaced_seq1.split()) + len(spaced_seq2.split()) + 2): #special tokens added by the tokenizer (T5)
    #     sentence_b_start = token_type_ids[0].tolist().index(1)
    #     print('tokenizer added')

    else:
        raise Exception("Special tokens added in an unfamiliar format")

    if verbose > 0:
        print("attention size:", len(attn), "x", attn[0].size())
        print(f"sequence 2 starts at: {sentence_b_start}")

    return attn


def zero_self_attn(attn, l1, l2):
    """This function zero outs the self attention (seq1->seq1 and seq2->seq2)
    l1 and l2 are the length of input. l1 and l2 are from seqeunces with the special tokens added.
    """
    temp_attn = attn
    with torch.no_grad():
        l = 0
        for layer in temp_attn:
            l += 1
            for head in layer.squeeze(0):
                head[:l1, :l1] = torch.zeros((l1, l1))
                head[l1:, l1:] = torch.zeros((l2, l2))
    return attn


def accuracy_torch(y_pred, y_true):
    """
    Calculates accuracy between two torch tensors.

    Parameters:
    - y_pred (torch.tensor): predicted labels
    - y_true (torch.tensor): true labels

    Returns:
    - accuracy (float): accuracy score
    """
    correct = (y_pred == y_true).sum().item()
    total = y_true.shape[0]
    accuracy = correct / total

    return accuracy


def f1_torch(y_true, y_pred):
    """
    Computes the F1 score for a given threshold.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Predicted labels.

    Returns
    -------
    f1 : torch.Tensor
        F1 score.
    precision : torch.Tensor
        Precision score.
    recall : torch.Tensor
        Recall score.
    """

    precision = precision_score(y_true.cpu().detach(), y_pred.cpu().detach(), average='weighted', zero_division=1)
    recall = recall_score(y_true.cpu().detach(), y_pred.cpu().detach(), average='weighted', zero_division=1)
    f1 = f1_score(y_true.cpu().detach(), y_pred.cpu().detach(), average='weighted', zero_division=1)

    f1, precision, recall = torch.tensor(f1), torch.tensor(precision), torch.tensor(recall)

    return f1, precision, recall


def get_best_run_wandb(sweep_id, wandb_proj):

    """
    Get the best run from a sweep and save the config to a json file

    Parameters
    ----------
    sweep_id : str
        the sweep id
    wandb_proj : str
        the wandb project name

    Returns
    -----------
    best_run : wandb.wandb_run.Run
        the best run
    
    """
    api = wandb.Api()
    print(sweep_id)
    try:
        sweep = api.sweep(f"{wandb_proj}/{sweep_id}")
    except Exception as e:
        print(e)
        sweep = api.sweep(f"AcrTransAct_v4.2/{sweep_id}")
    best_run = sweep.best_run()

    # os.makedirs(f"results/best_runs", exist_ok=True)
    # with open(f"results/best_runs/{sweep_id}.json", "w") as f:
    #     json.dump(best_run.config, f)

    wandb.finish()
    return best_run.config


def find_best_lr(model, trainer, loaders, min_lr= 3e-4, max_lr= 3e-3):
    """
    This function finds the best learning rate for a given model

    Parameters
    ----------
    model : pytorch_lightning.LightningModule

    trainer : pytorch_lightning.Trainer

    loaders : dict
        dictionary containing the train and validation dataloaders

    Returns
    -------
    float
        the best learning rate
    """
    lr_finder = trainer.tuner.lr_find(
    model,
    loaders["train_loader"],
    loaders["val_loader"],
    min_lr=min_lr,
    max_lr=max_lr,
    num_training=500,
    mode="exponential",
    )
    new_lr = lr_finder.suggestion()
    return new_lr

