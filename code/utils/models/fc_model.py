import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from utils.model_utils import f1_torch, accuracy_torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR


class AcrTransAct_FNN(pl.LightningModule):
    """
    A CNN model for inhibition prediction. It can use the amino acid sequence, the secondary structure or both.
    
        Params keys
        ----------
        monitor_metric_LR : str
            metric to monitor for early stopping and checkpointing
        lr : float
            learning rate. Initial learning rate can be set through this variable
        seq_len_aa, seq_len_ss : int
            number of input channels in the conv layer for esm and ss convs
        hidden_size : int
            size of the feature vector for each amino acid
        kernel_size : int
            size of the convolutional kernel
        out_channels : int
            number of output channels in the convolutional layer
        weight_decay : float
            weight decay for the optimizer
        dout : float
            dropout rate
        mode : str
            mode for the network. Can be 1, 2 or 3. each mode adds an extra FC layer to the network
        FC_nodes : int
            number of nodes in the FC layer
        optimizer : str
            optimizer to use. Can be Adam or SGD
        """

    def __init__(self, params, channels_first = True, debug_mode = False, verbose=False):
        
        super().__init__()

        self.debug_mode = debug_mode

        self.use_aa = params["use_aa"]
        self.use_ss = params["use_ss"]

        assert self.use_aa or self.use_ss, "At least one of the features must be used"

        self.monitor_metric_LR = params["monitor_metric_LR"]
        self.lr = params["lr"]
        self.channels_first = channels_first

        self.seq_len_aa = params["seq_len_aa"] if self.use_aa else 0
        self.seq_len_ss = params["seq_len_ss"] if self.use_ss else 0
        self.hidden_size_aa = params["hidden_size_aa"] if self.use_aa else 0
        self.kernel_size_aa = params["kernel_size_aa"] if self.use_aa else 0
        self.out_channels_aa = params["out_channels_aa"] if self.use_aa else 0
        self.hidden_size_ss = params["hidden_size_ss"] if self.use_ss else 0
        self.kernel_size_ss = params["kernel_size_ss"] if self.use_ss else 0
        self.out_channels_ss = params["out_channels_ss"] if self.use_ss else 0

        self.weight_decay = params["weight_decay"]
        self.dout = params["dout"]
        self.mode = params["mode"]
        self.FC_nodes = params["FC_nodes"]
        self.reduce_lr = params["reduce_lr"]

        if self.reduce_lr == "plateau":
            self.lr_reduce_factor = params["lr_reduce_factor"]

        self.optimizer_name = params["optimizer"]

        self.base_lr = params["base_lr"] if self.reduce_lr == "cyclic" else 0
        self.max_lr = params["max_lr"] if self.reduce_lr == "cyclic" else 0
        self.step_size_up = (
            params["step_size"] if self.reduce_lr == "cyclic" else 0
        )
        self.step_size_down = (
            params["step_size"] if self.reduce_lr == "cyclic" else 0
        )
        self.mode_cyclic = (
            params["mode_cyclic"] if self.reduce_lr == "cyclic" else "not used"
        )
        
        ##########################################################
        if self.use_aa:
            # self.conv_aa = nn.Conv1d(
            #     in_channels=self.hidden_size_aa if self.channels_first else self.seq_len_aa,
            #     out_channels=self.out_channels_aa,
            #     kernel_size=self.kernel_size_aa,
            # )
            self.fc_aa = nn.Linear(self.hidden_size_aa* self.seq_len_aa, self.out_channels_aa)
            torch.nn.init.xavier_uniform_(self.fc_aa.weight)
            self.bn1_aa = nn.BatchNorm1d(self.out_channels_aa)

        if self.use_ss:
            # self.conv_ss = nn.Conv1d(
            #     in_channels=self.hidden_size_ss if self.channels_first else self.seq_len_ss,
            #     out_channels=self.out_channels_ss,
            #     kernel_size=self.kernel_size_ss,
            # )
            self.fc_ss = nn.Linear(self.hidden_size_ss*self.seq_len_ss, self.out_channels_ss)
            torch.nn.init.xavier_uniform_(self.fc_ss.weight)
            self.bn1_ss = nn.BatchNorm1d(self.out_channels_ss)

        ##########################################################
        
        if self.channels_first: # channels first mode (batch, channels, seq_len)
            in_features_aa = self.out_channels_aa * (
                self.seq_len_aa# - self.kernel_size_aa + 1
            )  # if self.use_aa is False it will be 0
            in_features_ss = self.out_channels_ss * (
                self.seq_len_ss# - self.kernel_size_ss + 1
            )  # if self.use_ss is False it will be 0
        else: # channels last mode (batch, seq_len, channels)
            in_features_aa = self.out_channels_aa * (
                self.hidden_size_aa# - self.kernel_size_aa + 1
            )
            in_features_ss = self.out_channels_ss * (
                self.hidden_size_ss# - self.kernel_size_ss + 1
            )
     
        # if self.mode == 0:
        #     gap = nn.AdaptiveAvgPool1d(1)
        #     self.fc_last_layer0 = nn.Linear(
        #         in_features=in_features_aa + in_features_ss,
        #         out_features=1,
        #     )

        # last layer on mode 1
        if self.mode >= 1:
            self.fc1 = nn.Linear(
                in_features=self.out_channels_aa + self.out_channels_ss,
                out_features=1 if self.mode == 1 else self.FC_nodes,
            )
            torch.nn.init.xavier_uniform_(self.fc1.weight)

        if self.mode >= 2:
            # last layer on mode 2 and 3
            self.fc_last_layer2 = nn.Linear(in_features=self.FC_nodes, out_features=1)
            torch.nn.init.xavier_uniform_(self.fc_last_layer2.weight)
            self.bn2 = nn.BatchNorm1d(self.FC_nodes)

        if self.mode >= 3:
            self.fc2 = nn.Linear(in_features=self.FC_nodes, out_features=self.FC_nodes)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.dropout = nn.Dropout(self.dout)

        self.save_hyperparameters()  # save the paramters for logging with Wandb

        self.verbose = verbose

        # lists for saving the outputs of the training and validation steps
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # save the best f1 score for debugging the model # TODO: remove this
        self.best_f1 = 0
        self.best_epoch = 0

    def forward(self, x):
        if self.use_ss and self.use_aa:
            x_aa, x_ss = x[0], x[1]
        elif self.use_aa and not self.use_ss:
            x_aa = x
        elif self.use_ss and not self.use_aa:
            x_ss = x

        # CONV LAYER for ESM
        if self.use_aa:
            if self.debug_mode:
                print("1 x_aa shape before fc1: ", x_aa.shape)
            x_aa =  x_aa.view(-1, self.seq_len_aa * self.hidden_size_aa)
            x_aa = self.fc_aa(x_aa)#self.conv_aa(x_aa)

            if self.debug_mode:
                print("2 x_aa shape after conv: ", x_aa.shape)

            x_aa = self.bn1_aa(x_aa)
            x_aa = F.relu(x_aa)
            if self.dout != 0:
                x_aa = self.dropout(x_aa)
            
            x_aa = x_aa.view(x_aa.shape[0], -1)
            if self.debug_mode:
                print("3 x_aa shape after view: ", x_aa.shape)

        # CONV LAYER for SS
        if self.use_ss:
            if self.debug_mode:    
                print("1 x_ss shape before conv: ", x_ss.shape)
            x_ss =  x_ss.view(-1, self.seq_len_ss * self.hidden_size_ss)

            x_ss = self.fc_ss(x_ss)#self.conv_ss(x_ss)
            
            if self.debug_mode:
                print("2 x_ss shape after conv: ", x_ss.shape)

            x_ss = self.bn1_ss(x_ss)
            x_ss = F.relu(x_ss)
            if self.dout != 0:
                x_ss = self.dropout(x_ss)
            
            x_ss = x_ss.view(x_ss.shape[0], -1)
            if self.debug_mode:
                print("3 x_ss shape after view: ", x_ss.shape)

        if self.use_ss and self.use_aa:
            # concat the two conv layers outputs
            if self.debug_mode:
                print("4 x before concat ss and aa shape: ", x_aa.shape, x_ss.shape)

            x = torch.cat((x_aa, x_ss), dim=1)
            x = x.view(x.shape[0], -1)

            if self.debug_mode:
                print("5 x concat ss and aa shape: ", x.shape)

        elif self.use_aa and not self.use_ss:
            x = x_aa
        
        elif self.use_ss and not self.use_aa:
            x = x_ss

        # mode 0 global average pooling
        # if self.mode == 0:
        #     print("x shape before global average pooling: ", x.shape)
        #     x = self.gap(x)
        #     print("x shape after global average pooling: ", x.shape)

        # mode 1
        if self.mode == 1:
            if self.debug_mode:
                print("6 x shape before fc_last_layer1: ", x.shape)
            x = self.fc1(x)
            if self.debug_mode:
                print("7 x shape after fc_last_layer1: ", x.shape)

        # mode 2
        elif self.mode >= 2:
            
            if self.debug_mode:
                print("8 x shape before fc1: ", x.shape)
            x = self.fc1(x)
            x = self.bn2(x)
            x = F.relu(x)
            if self.dout != 0:
                x = self.dropout(x)
            if self.debug_mode:    
                print("9 x shape after fc1: ", x.shape)
            # mode 3
            if self.mode == 3:
                x = self.fc2(x)
                x = self.bn2(x)
                x = F.relu(x)
                if self.dout != 0:
                    x = self.dropout(x)

            x = self.fc_last_layer2(x)

        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)

        accuracy = (y_hat.round() == y).float().mean()

        f1 = f1_torch(y, y_hat)[0]
        logs = {"loss": loss, "accuracy": accuracy, "f1": f1}

        self.training_step_outputs.append(logs)

        return logs

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["accuracy"] for x in self.training_step_outputs]
        ).mean()
        avg_f1 = torch.stack([x["f1"] for x in self.training_step_outputs]).mean()

        if self.verbose == 1:
            print(
                f"- Training:  Loss = {avg_loss:.4f} | Accuracy = {avg_acc:.4f} | F1 Score = {avg_f1:.4f}"
            )
            print("*" * 60)

        self.log("train_loss", avg_loss)
        self.log("train_accuracy", avg_acc)
        self.log("train_f1_score", avg_f1)
        self.log("Learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        val_loss = F.binary_cross_entropy(y_hat, y)
        val_accuracy = (y_hat.round() == y).float().mean()

        f1 = f1_torch(y, y_hat)[0]

        logs = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_score": f1,
        }

        self.validation_step_outputs.append(logs)

        return logs

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_val_acc = torch.stack(
            [x["val_accuracy"] for x in self.validation_step_outputs]
        ).mean()
        avg_val_f1 = torch.stack(
            [x["val_f1_score"] for x in self.validation_step_outputs]
        ).mean()

        if self.verbose == 1:
            print(f"> Epoch {self.current_epoch+1}, best F1_val {self.best_f1:.3f} at epoch {self.best_epoch+1}, lr = {self.trainer.optimizers[0].param_groups[0]['lr']:.6f}")
            print(
                f"- Val: Loss = {avg_val_loss:.4f} | Accuracy = {avg_val_acc:.4f}| F1 Score = {avg_val_f1:.4f}"
            )

        # these logs are used for checkpointing based on
        # validation metrics and for vizualization with wandb
        self.log("val_loss", avg_val_loss)
        self.log("val_accuracy", avg_val_acc)
        self.log("val_f1_score", avg_val_f1)

        self.validation_step_outputs.clear()

        # save the new f1 score if it is better than the previous one
        if avg_val_f1.cpu().numpy() > self.best_f1:
            self.best_f1 = avg_val_f1.cpu().numpy()
            self.best_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        test_loss = F.binary_cross_entropy(y_hat, y)
        test_accuracy = (y_hat.round() == y).float().mean()

        f1 = f1_torch(y, y_hat)[0]

        logs = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_f1_score": f1,
        }

        self.test_step_outputs.append(logs)

        return logs

    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(
            [x["test_loss"] for x in self.test_step_outputs]
        ).mean()
        avg_test_acc = torch.stack(
            [x["test_accuracy"] for x in self.test_step_outputs]
        ).mean()
        avg_test_f1 = torch.stack(
            [x["test_f1_score"] for x in self.test_step_outputs]
        ).mean()

        if self.verbose == 1:
            print(
                f"- Test: Loss = {avg_test_loss:.4f} | Accuracy = {avg_test_acc:.4f}| F1 Score = {avg_test_f1:.4f}"
            )

        self.log("test_loss", avg_test_loss)
        self.log("test_accuracy", avg_test_acc)
        self.log("test_f1_score", avg_test_f1)

        self.test_step_outputs.clear()

        return {
            "test_loss": avg_test_loss,
            "test_accuracy": avg_test_acc,
            "test_f1_score": avg_test_f1,
        }

    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if self.reduce_lr == "plateau":
             # default mode for accuracy
            if self.monitor_metric_LR.lower().find("f1") != -1 or self.monitor_metric_LR.lower().find("acc") != -1:
                mode = "max"
            else:
                mode = "min" 

            if self.verbose:
                print(f"% ReduceLR mode: {mode}, monitor: {self.monitor_metric_LR}, factor: {self.lr_reduce_factor}")

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=self.lr_reduce_factor,
                patience=10,
                verbose=False,
                threshold=0.01 if mode == "min" else 0.001,
                min_lr=1e-8,
            )

        elif self.reduce_lr == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.base_lr,
                max_lr=self.max_lr,
                step_size_up=self.step_size_up,
                step_size_down=self.step_size_down,
                mode=self.mode_cyclic,
                cycle_momentum=False,
            )

        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler ,
                "monitor": self.monitor_metric_LR,
                "name": self.reduce_lr,
            },
        }

    # def eval_best_chkpt(self, X_val, y_val, chkpt_path, val_info=None, conf_mat=False):
    #     """
    #     Evaluate model on validation set

    #     Parameters
    #     ----------
    #     model : torch model
    #         model to be evaluated
    #     X_val : torch tensor
    #         validation set
    #     y_val : torch tensor
    #         validation labels
    #     model_path : str
    #         path to model checkpoint

    #     Returns
    #     -------
    #     dict
    #         dictionary with f1 score, precision, recall and accuracy
    #     """

    #     if conf_mat:
    #         cm = confusion_matrix(y_val, y_pred)
    #         sns.heatmap(cm, annot=True, fmt="d")
    #         plt.title("Confusion matrix for test set")
    #         plt.show()

    #     results = {
    #         "best_val_f1": f1,
    #         "best_val_accuracy": acc,
    #         "best_val_precision": precision,
    #         "best_val_recall": recall,
    #     }

    #     # get indices of misclassified samples
    #     if val_info is not None:
    #         indicies = np.where(y_val != y_pred)[0]
    #         mis_cls_samples = [val_info[i][0] for i in indicies]
    #         results["misclassified_samples"] = mis_cls_samples

    #     return results
