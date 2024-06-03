import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import seml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from sacred import Experiment
from torch_geometric.seed import seed_everything

import data.data
import data.data_ood
import wandb
from config import CHECKPOINT_DIR
from models.fragGNN import FragGNN, FragGNNSmall
from models.gcn import GCN, GCNSubstructure, VerySimpleGCN
from models.lightning_models import *

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    @ex.capture(prefix="data")
    def init_dataset(self,
                     _config,
                     dataset: str,
                     seed: Optional[int] = None,
                     remove_node_features: bool = False,
                     one_hot_degree: bool = False,
                     one_hot_node_features: bool = False,
                     one_hot_edge_features: bool = False,
                     fragmentation_method: Optional[Tuple[str, str,
                                                          Dict]] = None,
                     loader_params: Optional[Dict] = None,
                     encoding: List = [],
                     dataset_params={}):
        """Initialize train, validation and test loader.

        Parameters
        ----------
        dataset
            Name of the dataset
        seed
            Seed for everything
        remove_node_features, optional
            Boolean indicating whether node_labels should be removed, by default False
        one_hot_degree, optional
            Boolean indicating whether to concatinate the node features with a one hot encoded node degree, by default False.
        fragmentation_method, optional
            Tuple ``(name_of_fragmentation, type_of_fragmentation, vocab_size)``.
        loader_params, optional
            Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.
        encoding, optional
            List of encodings that should be used.
        dataset_params, optional
            If subset_frac in dataset_params: Only subset_frac of the dataset will be used for training.
            If filter in dataset_params: Only molecules containing no ring of size filter will be used for training.
            If higher_edge_features in dataset_params: Information about the edges in the higher level graph will be computed.
            If dataset_seed in dataset_params: Seperate seed for the dataset split.
        """
        print(f"Dataset received config: {_config}")
        if seed is not None:
            # torch.manual_seed(seed)
            seed_everything(seed)

        if fragmentation_method:
            self.num_substructures = fragmentation_method[2]["vocab_size"]
        else:
            self.num_substructures = None

        if "filter" in dataset_params:
            # only used in the ood experiment
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = data.data_ood.load_fragmentation(
                dataset,
                remove_node_features=remove_node_features,
                one_hot_degree=one_hot_degree,
                one_hot_node_features=one_hot_node_features,
                one_hot_edge_features=one_hot_edge_features,
                fragmentation_method=fragmentation_method,
                loader_params=loader_params,
                **dataset_params)
        else:
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = data.data.load_fragmentation(
                dataset,
                remove_node_features=remove_node_features,
                one_hot_degree=one_hot_degree,
                one_hot_node_features=one_hot_node_features,
                one_hot_edge_features=one_hot_edge_features,
                fragmentation_method=fragmentation_method,
                loader_params=loader_params,
                encoding=encoding,
                **dataset_params)

    @ex.capture(prefix="model")
    def init_model(self,
                   model_type: str,
                   model_params: dict,
                   classification: bool = True):
        self.classification = classification
        model_params = model_params.copy()  # allows us to add fields to it
        if not "out_channels" in model_params:
            if classification:
                model_params[
                    "out_channels"] = self.num_classes if self.num_classes > 2 else 1
            else:
                model_params["out_channels"] = self.num_classes
        model_params["in_channels"] = self.num_features
        if model_type == "GCN":
            self.model = GCN(**model_params)
        elif model_type == "VerySimpleGCN":
            self.model = VerySimpleGCN(**model_params)
        elif model_type == "GCNSubstructure":
            model_params["in_channels_substructure"] = self.num_substructures
            self.model = GCNSubstructure(**model_params)
        elif model_type == "FragGNNSmall":
            model_params["in_channels_substructure"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  # TODO: could be different for other datasets
            self.model = FragGNNSmall(**model_params)
        elif model_type == "FragGNN":
            model_params["in_channels_substructure"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  # TODO: could be different for other datasets
            self.model = FragGNN(**model_params)
        else:
            raise RuntimeError(f"Model {model_type} not supported")
        print("Setup model:")
        print(self.model)

    @ex.capture(prefix="optimization")
    def init_optimizer(self,
                       optimization_params,
                       scheduler_parameters=None,
                       loss: Optional[str] = None,
                       additional_metric: Optional[str] = None,
                       ema_decay=None):
        loss_func = None
        if self.classification and self.num_classes > 2:
            loss_func = ce_loss
            acc = classification_accuracy
        elif self.classification:
            loss_func = bce_loss
            acc = binary_classification_accuracy
        else:
            if loss and loss == "mae":
                loss_func = mae_loss
            else:
                loss_func = mse_loss
            acc = regression_acc

        additional_metric_func = None
        if additional_metric:
            if additional_metric == "mae":
                additional_metric_func = mae_loss
            elif additional_metric == "mse":
                additional_metric_func = mse_loss
            elif additional_metric == "auroc":
                additional_metric_func = auroc
            elif additional_metric == "ap":
                additional_metric_func = average_multilabel_precision
            elif additional_metric == "counting_experiment":
                additional_metric_func = [
                    regression_acc, regression_precision, regression_recall, num_true_positives]

        self.lightning_model = LightningModel(
            model=self.model,
            loss=loss_func,
            acc=acc,
            optimizer_parameters=optimization_params,
            scheduler_parameters=scheduler_parameters,
            additional_metric=additional_metric_func,
            ema_decay=ema_decay)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    @ex.capture()
    def train(self, trainer_params, project_name, _config, notes="", ckpt_path=None, use_wandb=True):

        checkpoint_directory = f"{CHECKPOINT_DIR}/{_config['db_collection']}/run-{_config['overwrite']}"
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)

        if use_wandb:
            wandb_logger = WandbLogger(
                name=f"{_config['db_collection']}_{_config['overwrite']}",
                project=project_name,
                save_dir=checkpoint_directory,
                notes=notes,
                entity="frags")
            wandb_logger.experiment.config.update(_config)
        else:
            wandb_logger = None

        if "gradient_clip_val" in trainer_params:
            additional_params = {
                "gradient_clip_val": trainer_params["gradient_clip_val"]
            }
        else:
            additional_params = {}

        monitor = trainer_params["monitor"] if "monitor" in trainer_params else "val_loss"
        mode = "min" if monitor == "val_loss" else "max"
        patience = trainer_params["patience_early_stopping"] if "patience_early_stopping" in trainer_params else 50

        if "min_lr" in trainer_params:
            trainer = Trainer(
                max_epochs=trainer_params["max_epochs"],
                logger=wandb_logger,
                log_every_n_steps=15,
                default_root_dir=checkpoint_directory,
                detect_anomaly=True,
                callbacks=[
                    EarlyStopping(monitor=monitor,
                                  mode=mode,
                                  patience=patience,
                                  verbose=True),
                    ModelCheckpoint(monitor=monitor, mode=mode)
                ],
                enable_progress_bar=False,
                **additional_params)
        else:
            trainer = Trainer(
                max_epochs=trainer_params["max_epochs"],
                logger=wandb_logger,
                log_every_n_steps=15,
                default_root_dir=checkpoint_directory,
                detect_anomaly=True,
                callbacks=[
                    EarlyStopping(monitor=monitor,
                                  mode=mode,
                                  patience=patience,
                                  verbose=True),
                    ModelCheckpoint(monitor=monitor, mode=mode)
                ],
                enable_progress_bar=False,
                **additional_params)

        trainer.fit(self.lightning_model,
                    train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader,
                    ckpt_path=ckpt_path)
        if trainer_params["testing"] == True:
            result = trainer.test(self.lightning_model, self.test_loader)
            print(f"Test result: {result}")
            if use_wandb:
                wandb.finish()
            return result
        else:
            if use_wandb:
                wandb.finish()


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
