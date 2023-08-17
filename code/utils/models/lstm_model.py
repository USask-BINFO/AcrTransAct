import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from utils.model_utils import f1_torch


class AcrTransAct_LSTM(pl.LightningModule):
    """
    A CNN model for inhibition prediction. It can use the amino acid sequence, the secondary structure or both.

        Params keys
        ----------
        monitor_metric_LR : str
            metric to monitor for early stopping and checkpointing
        lr : float
            learning rate. Initial learning rate can be set through this variable
        seq_len_aa, seq_len_sf : int
            number of input channels in the conv layer for esm and ss convs
        hidden_size : int
            size of the feature vector for each amino acid
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

    def __init__(self, params, channels_first=True, debug_mode=False, verbose=False):
        super().__init__()

        self.debug_mode = debug_mode
        self.verbose = verbose

        self.use_aa = params["use_aa"]
        self.use_sf = params["use_sf"]

        assert self.use_aa or self.use_sf, "At least one of the features must be used"

        self.monitor_metric_lr = params["monitor_metric_lr"]
        self.lr = params["lr"]
        self.channels_first = channels_first
        self.seq_len_aa = params["seq_len_aa"] if self.use_aa else 0
        self.seq_len_sf = params["seq_len_sf"] if self.use_sf else 0
        self.hidden_size_aa = params["hidden_size_aa"] if self.use_aa else 0
        self.out_channels_aa = params["out_channels_aa"] if self.use_aa else 0
        self.hidden_size_sf = params["hidden_size_sf"] if self.use_sf else 0
        self.out_channels_sf = params["out_channels_sf"] if self.use_sf else 0
        self.weight_decay = params["weight_decay"]
        self.dout = params["dout"]
        self.mode = params["mode"]
        self.FC_nodes = params["FC_nodes"]
        self.reduce_lr = params["reduce_lr"]
        if self.reduce_lr == "plateau":
            self.lr_reduce_factor = params["lr_reduce_factor"]

        self.optimizer_name = params["optimizer"]
        self.class_weights = params["class_weights"]

        if self.reduce_lr == "cyclic":
            self.base_lr = params["base_lr"] 
            self.max_lr = params["max_lr"] 
            self.step_size_up = params["step_size"] 
            self.step_size_down = params["step_size"] 
            self.mode_cyclic = params["mode_cyclic"] 

        self.num_layers = params["num_main_layers"]  # determines the number of LSTM layers

        self.train_loss_history, self.train_acc_history, self.train_f1_history = (
            [],
            [],
            [],
        )
        self.val_loss_history, self.val_acc_history, self.val_f1_history = [], [], []

        # lists for saving the outputs of the training and validation steps
        (
            self.training_step_outputs,
            self.validation_step_outputs,
            self.test_step_outputs,
        ) = ([], [], [])

        # save the best metric for hyperparam search
        if "optimize_metric" in params.keys():
            self.optimize_metric = params["optimize_metric"]
            self.val_loss_best_metric = np.inf
            self.val_acc_best_metric = 0
            self.val_f1_best_metric = 0
            self.best_epoch = 1
        else:
            self.optimize_metric = None

        ########################### LSTM ###############################
        if self.use_aa:
            self.lstm_aa = nn.LSTM(
                input_size=self.hidden_size_aa,
                hidden_size=self.out_channels_aa,
                batch_first=True,
                bidirectional=True,
                num_layers=self.num_layers,
                dropout=self.dout,
            )
            self.bn1_aa = nn.BatchNorm1d(self.seq_len_aa)

        if self.use_sf:
            self.lstm_sf = nn.LSTM(
                input_size=self.hidden_size_sf,
                hidden_size=self.out_channels_sf,
                batch_first=True,
                bidirectional=True,
                num_layers=self.num_layers,
                dropout=self.dout,
            )
            self.bn1_sf = nn.BatchNorm1d(self.seq_len_sf)
        ##########################################################
        # channels first mode (batch, channels, seq_len)
        if self.channels_first:
            in_features_aa = (
                self.out_channels_aa * self.seq_len_aa * 2 if self.use_aa else 0
            )  # *2 because of bidirectional
            in_features_sf = (
                self.out_channels_sf * self.seq_len_sf * 2 if self.use_sf else 0
            )  # *2 because of bidirectional

        # channels last mode (batch, seq_len, channels)
        else:
            in_features_aa = (
                self.out_channels_aa * self.hidden_size_aa if self.use_aa else 0
            )
            in_features_sf = (
                self.out_channels_sf * self.hidden_size_sf if self.use_sf else 0
            )
        ##########################################################
        # last layer on mode 1
        if self.mode >= 1:
            self.fc1 = nn.Linear(
                in_features=in_features_aa + in_features_sf,
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

        # lists for saving the outputs of the training and validation steps
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # # save the best f1 score for debugging the model # TODO: remove this
        # self.best_f1 = 0
        # self.best_val_loss = None
        # self.best_epoch = 0

    def forward(self, x):
        if self.use_sf and self.use_aa:
            x_aa, x_sf = x[0].to(self.device), x[1].to(self.device)
            if self.channels_first:
                x_aa = x_aa.permute(0, 2, 1)
                x_sf = x_sf.permute(0, 2, 1)
        elif self.use_aa and not self.use_sf:
            x_aa = x.to(self.device)
            if self.channels_first:
                x_aa = x_aa.permute(0, 2, 1)
        elif self.use_sf and not self.use_aa:
            x_sf = x.to(self.device)
            if self.channels_first:
                x_sf = x.permute(0, 2, 1)

        # LSTM layer for ESM
        if self.use_aa:
            if self.debug_mode:
                print("1 x_aa shape before LSTM: ", x_aa.shape)
            x_aa, _ = self.lstm_aa(x_aa)
            if self.debug_mode:
                print("2 x_aa shape after LSTM: ", x_aa.shape)

            x_aa = self.bn1_aa(x_aa)
            x_aa = F.relu(x_aa)
            if self.dout != 0:
                x_aa = self.dropout(x_aa)

            x_aa = x_aa.view(x_aa.shape[0], -1)
            if self.debug_mode:
                print("3 x_aa shape after view: ", x_aa.shape)

        # LSTM layer for SS
        if self.use_sf:
            if self.debug_mode:
                print("1 x_sf shape before LSTM: ", x_sf.shape)
            x_sf, _ = self.lstm_sf(x_sf)
            if self.debug_mode:
                print("2 x_sf shape after LSTM: ", x_sf.shape)

            x_sf = self.bn1_sf(x_sf)
            x_sf = F.relu(x_sf)
            if self.dout != 0:
                x_sf = self.dropout(x_sf)
            if self.debug_mode:
                print("3 x_sf shape after dropout: ", x_sf.shape)
            x_sf = x_sf.view(x_sf.shape[0], -1)
            if self.debug_mode:
                print("4 x_sf shape after view: ", x_sf.shape)

        if self.use_aa and self.use_sf:
            x = torch.cat((x_aa, x_sf), dim=1)
            if self.debug_mode:
                print("5 x concat ss and aa shape: ", x.shape)
            x = x.view(x.shape[0], -1)
            if self.debug_mode:
                print("6 x shape after view: ", x.shape)

        elif self.use_aa and not self.use_sf:
            x = x_aa

        elif self.use_sf and not self.use_aa:
            x = x_sf

        # mode 1
        if self.mode >= 1:
            if self.debug_mode:
                print("6 x shape before fc1: ", x.shape)
            x = self.fc1(x)
            if self.debug_mode:
                print("7 x shape after fc1: ", x.shape)
            # mode 2
            if self.mode >= 2:
                x = self.bn2(x)
                x = F.relu(x)
                if self.dout != 0:
                    x = self.dropout(x)
                if self.debug_mode:
                    print("7 x shape after fc1: ", x.shape)
                # mode 3
                if self.mode == 3:
                    x = self.fc2(x)
                    x = self.bn2(x)
                    x = F.relu(x)
                    if self.dout != 0:
                        x = self.dropout(x)

                x = self.fc_last_layer2(x)

        x = torch.sigmoid(x)
        # x = torch.nn.functional.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        y = y.view(-1).to(self.device)

        sample_weights = torch.where(y > 0.5, self.class_weights[1], self.class_weights[0]).to(self.device)
        loss = F.binary_cross_entropy(y_hat, y, weight=sample_weights)
        
        accuracy = (y_hat.round() == (y > 0.5).type(torch.LongTensor).to(self.device)).float().mean()
        f1 = f1_torch((y > 0.5).type(torch.LongTensor).to(self.device), y_hat.round())[0]

        logs = {"loss": loss, "accuracy": accuracy, "f1": f1}
        self.log("train_loss", loss, prog_bar=True)

        self.training_step_outputs.append(logs)

        return logs


    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack(
            [x["accuracy"] for x in self.training_step_outputs]
        ).mean()
        avg_f1 = torch.stack([x["f1"] for x in self.training_step_outputs]).mean()

        if self.verbose == 1 and self.current_epoch % 10 == 0:
            print(
                f"- Training:  Loss = {avg_loss:.4f} | Accuracy = {avg_acc:.4f} | F1 Score = {avg_f1:.4f}"
            )
            print("*" * 60)

        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_accuracy", avg_acc, prog_bar=True)
        self.log("train_f1_score", avg_f1, prog_bar=True)
        self.log("lr_rate", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.training_step_outputs.clear()

        # for plotting outside of wandb
        with torch.no_grad():
            self.train_loss_history.append(avg_loss.detach().cpu().numpy())
            self.train_acc_history.append(avg_acc.detach().cpu().numpy())
            self.train_f1_history.append(avg_f1.detach().cpu().numpy())


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        y = y.view(-1).to(self.device)

        sample_weights = torch.where(y > 0.5, self.class_weights[1], self.class_weights[0]).to(self.device)
        loss = F.binary_cross_entropy(y_hat, y, weight=sample_weights)
        
        accuracy = (y_hat.round() == (y > 0.5).type(torch.LongTensor).to(self.device)).float().mean()
        f1 = f1_torch((y > 0.5).type(torch.LongTensor).to(self.device), y_hat.round())[0]

        logs = {
            "val_loss": loss,
            "val_accuracy": accuracy,
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
        self.validation_step_outputs.clear()

        if self.verbose and self.current_epoch % 20 == 0:
            print(
                f"> Epoch {self.current_epoch+1}| best f1_val {self.best_val_f1:.2f},"+\
                f" val_loss at best f1: {self.best_f1_val_loss:.2f} epoch {self.best_epoch+1}"+\
                f"| lr = {self.trainer.optimizers[0].param_groups[0]['lr']:.6f}"
            )
            print(
                f"- Val: Loss = {avg_val_loss:.2f} | Accuracy = {avg_val_acc:.2f}| F1 Score = {avg_val_f1:.2f}"
            )

        # these logs are used for the logger wandb/tflogger
        self.log("val_loss", avg_val_loss, prog_bar=True)
        self.log("val_accuracy", avg_val_acc, prog_bar=True)
        self.log("val_f1_score", avg_val_f1, prog_bar=True)

        # for plotting without wandb
        self.val_loss_history.append(avg_val_loss.cpu().numpy())
        self.val_f1_history.append(avg_val_f1.cpu().numpy())
        self.val_acc_history.append(avg_val_acc.cpu().numpy())

        if self.optimize_metric.lower().__contains__("loss") and \
            avg_val_loss.cpu().numpy() < self.val_loss_best_metric:
                self.best_epoch = self.current_epoch + 1
                self.val_acc_best_metric = avg_val_acc.cpu().numpy()
                self.val_f1_best_metric = avg_val_f1.cpu().numpy()
                self.val_loss_best_metric = avg_val_loss.cpu().numpy()

        elif self.optimize_metric.lower().__contains__("f1") and \
            avg_val_f1.cpu().numpy() > self.val_f1_best_metric:
                self.best_epoch = self.current_epoch + 1
                self.val_acc_best_metric = avg_val_acc.cpu().numpy()
                self.val_loss_best_metric = avg_val_loss.cpu().numpy()
                self.val_f1_best_metric = avg_val_f1.cpu().numpy()


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1).to(self.device)
        y = y.view(-1).to(self.device)

        sample_weights = torch.where(y > 0.5, self.class_weights[1], self.class_weights[0]).to(self.device)
        loss = F.binary_cross_entropy(y_hat, y, weight=sample_weights)
        
        accuracy = (y_hat.round() == (y > 0.5).type(torch.LongTensor).to(self.device)).float().mean()
        f1, precision, recall = f1_torch((y > 0.5).type(torch.LongTensor).to(self.device), y_hat.round())

        logs = {
            "test_loss": loss,
            "test_acc": accuracy,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
        }
        self.test_step_outputs.append(logs)
        return logs


    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(
            [x["test_loss"] for x in self.test_step_outputs]
        ).mean()
        avg_test_acc = torch.stack(
            [x["test_acc"] for x in self.test_step_outputs]
        ).mean()
        avg_test_f1 = torch.stack(
            [x["test_f1"] for x in self.test_step_outputs]
        ).mean()
        avg_test_precision = torch.stack(
            [x["test_precision"] for x in self.test_step_outputs]
        ).mean()
        avg_test_recall = torch.stack(
            [x["test_recall"] for x in self.test_step_outputs]
        ).mean()

        if self.verbose:
            print(
                f"- Test: Loss = {avg_test_loss:.4f} | Accuracy = {avg_test_acc:.4f}| F1 Score = {avg_test_f1:.4f}"
            )

        # needed for returning the results 
        self.log("test_loss", avg_test_loss)
        self.log("test_acc", avg_test_acc)
        self.log("test_f1", avg_test_f1)
        self.log("test_precision", avg_test_precision)
        self.log("test_recall", avg_test_recall)

        self.test_step_outputs.clear()

        return {
            "test_loss": avg_test_loss,
            "test_acc": avg_test_acc,
            "test_f1": avg_test_f1,
            "test_precision": avg_test_precision,
            "test_recall": avg_test_recall,
        }


    def pred_val(self, val_loader, use_aa, use_sf, return_probs=False):
        """
        Predicts on validation set

        Parameters
        ----------
        val_loader : pytorch dataloader
            validation set

        Returns
        -------
        preds : numpy array
            predictions on validation set (either classes or probabilities)
        """
        with torch.no_grad():
            self.eval()
            preds = []
            for item in val_loader:
                if use_aa and use_sf:
                    features = item[0][0].to("cpu"), item[0][1].to("cpu")
                else:
                    features = item[0].to("cpu")
                pred = self(features)

                preds.append(pred)
            preds = np.concatenate(preds)
            
        return preds


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
            if (
                self.monitor_metric_lr.lower().find("f1") != -1
                or self.monitor_metric_lr.lower().find("acc") != -1
            ):
                mode = "max"
            else:
                mode = "min"

            if self.verbose:
                print(
                    f"% ReduceLR mode: {mode}, monitor: {self.monitor_metric_lr}, factor: {self.lr_reduce_factor}"
                )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=self.lr_reduce_factor,
                patience=10,
                verbose=False,
                threshold=0.01 if mode == "min" else 0.001,
                min_lr=1e-6,
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
                "scheduler": scheduler,
                "monitor": self.monitor_metric_lr,
                "name": self.reduce_lr,
            },
        }



    def plot_history(self, save_path=None, running_mean_window=1, plot=False):
        """
        Plot training and validation history in 3 different subplots for F1 score, accuracy, and loss.
        Apply a running mean to every 3 epochs to smooth the curves. and indicates the best epoch by an x

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot, by default None
        running_mean_window : int, optional
            Window size for the running mean, by default 1
        best_metric : str, optional
            Metric to use to determine the best epoch, by default "f1"
            options: "f1", "acc", "loss"
        """

        _, ax = plt.subplots(3, 1, figsize=(10, 10))

        train_f1_smooth = np.convolve(self.train_f1_history, np.ones(running_mean_window), 'valid') / running_mean_window
        val_f1_smooth = np.convolve(self.val_f1_history, np.ones(running_mean_window), 'valid') / running_mean_window
        train_loss_smooth = np.convolve(self.train_loss_history, np.ones(running_mean_window), 'valid') / running_mean_window
        val_loss_smooth = np.convolve(self.val_loss_history, np.ones(running_mean_window), 'valid') / running_mean_window
        train_acc_smooth = np.convolve(self.train_acc_history, np.ones(running_mean_window), 'valid') / running_mean_window
        val_acc_smooth = np.convolve(self.val_acc_history, np.ones(running_mean_window), 'valid') / running_mean_window

        best_epoch_f1 = np.argmax(val_f1_smooth)
        best_val_f1= np.max(val_f1_smooth)
        best_epoch_acc = np.argmax(val_acc_smooth)
        best_val_acc= np.max(val_acc_smooth)
        best_epoch_loss = np.argmin(val_loss_smooth)
        best_val_loss= np.min(val_loss_smooth)
        
        ax[0].plot(train_f1_smooth, label="Training")
        ax[0].plot(val_f1_smooth, label="Validation")
        ax[0].plot(best_epoch_f1, val_f1_smooth[best_epoch_f1], "x", label=f"Best f1 epoch")
        ax[0].set_title("F1 Score")
        ax[0].legend()

        ax[1].plot(train_acc_smooth, label="Training")
        ax[1].plot(val_acc_smooth, label="Validation")
        ax[1].plot(best_epoch_acc, val_acc_smooth[best_epoch_acc], "x", label=f"Best acc epoch")
        ax[1].set_title("Accuracy")
        ax[1].legend()

        ax[2].plot(train_loss_smooth, label="Training")
        ax[2].plot(val_loss_smooth, label="Validation")
        ax[2].plot(best_epoch_loss, val_loss_smooth[best_epoch_loss], "x", label=f"Best loss epoch")
        ax[2].set_title("Loss")
        ax[2].legend()

        print(f"Best f1 : {best_val_f1:.3f} at epoch {best_epoch_f1}")
        print(f"Best acc : {best_val_acc:.3f} at epoch {best_epoch_acc}")
        print(f"Best loss : {best_val_loss:.3f} at epoch {best_epoch_loss}")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if plot:
            plt.show()
