from sacred import Experiment
import numpy as np
import seml
from typing import List, Dict, Optional, Tuple
import datasets.data
# import datasets.data_ood
from models.gcn import GCN, VerySimpleGCN, GCNSubstructure, HimpNet, HimpNetHigherGraph, HimpNetAlternative
from models.hlg import HLG, HLGAlternative, HLG_Old, HLG_HIMP
from models.substructure_model import SubstructureNeuralNet
from models.pool_linear import PoolLinear, GlobalLinear
from models.lightning_models import *
from models.lightning_progress_bar import MeterlessProgressBar

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import wandb
import os

checkpoint_dir = "/ceph/hdd/students/kempern/substructure-gnns/models/checkpoints/checkpoints"
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

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
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
        encoding_list, optional
            List of encodings that should be used.
        dataset_params, optional
            If subset_frac in dataset_params: Only subset_fracof the dataset will be used for training.
            If filter in dataset_params: Only molecules containing no ring of size filter will be used for training.
            If higher_edge_features in dataset_params: Information about the edges in the higher level graph will be computed.
            If dataset_seed in dataset_params: Seperate seed for the dataset split.
        """
        print(f"Dataset received config: {_config}")
        if seed:
            torch.manual_seed(seed)

        if fragmentation_method:
            self.num_substructures = fragmentation_method[2]["vocab_size"]
        else:
            self.num_substructures = None

        if "filter" in dataset_params:
            # ood experiment
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = datasets.data_ood.load_fragmentation(
                dataset,
                remove_node_features=remove_node_features,
                one_hot_degree=one_hot_degree,
                one_hot_node_features=one_hot_node_features,
                one_hot_edge_features=one_hot_edge_features,
                fragmentation_method=fragmentation_method,
                loader_params=loader_params,
                **dataset_params)
        else:
            self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes = datasets.data.load_fragmentation(
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
        elif model_type == "SubstrucutureNet":
            model_params["num_substructures"] = self.num_substructures
            self.model = SubstructureNeuralNet(**model_params)
        elif model_type == "VerySimpleGCN":
            self.model = VerySimpleGCN(**model_params)
        elif model_type == "PoolLinear":
            self.model = PoolLinear(**model_params)
        elif model_type == "GlobalLinear":
            model_params.pop("in_channels")  #does not need in_channels
            self.model = GlobalLinear(**model_params)
        elif model_type == "GCNSubstructure":
            model_params["in_channels_substructure"] = self.num_substructures
            self.model = GCNSubstructure(**model_params)
        elif model_type == "HimpNet":
            model_params["in_channels_substructure"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HimpNet(**model_params)
        elif model_type == "HimpNetHigherGraph":
            model_params["in_channels_substructure"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HimpNetHigherGraph(**model_params)
        elif model_type == "HimpNetAlternative":
            model_params["in_channels_substructure"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HimpNetAlternative(**model_params)
        elif model_type == "HLG":
            model_params["in_channels_frag"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HLG(**model_params)
        elif model_type == "HLGAlternative":
            model_params["in_channels_frag"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HLGAlternative(**model_params)
        elif model_type == "HLG_Old":
            model_params["in_channels_frag"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HLG_Old(**model_params)
        elif model_type == "HLG_HIMP":
            model_params["in_channels_frag"] = self.num_substructures
            model_params[
                "in_channels_edge"] = 4  #TODO: could be different for other datasets
            self.model = HLG_HIMP(**model_params)
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
    def train(self, trainer_params, project_name, _config, notes=""):

        checkpoint_directory = f"{checkpoint_dir}/{_config['db_collection']}/run-{_config['overwrite']}"
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)

        wandb_logger = WandbLogger(
            name=f"{_config['db_collection']}_{_config['overwrite']}",
            project=project_name,
            save_dir=checkpoint_directory,
            notes=notes,
            entity="frags")
        wandb_logger.experiment.config.update(_config)
        #wandb_logger.watch(self.lightning_model, log="all")

        # bar = MeterlessProgressBar() # progress bar without a running bar

        if "gradient_clip_val" in trainer_params:
            additional_params = {
                "gradient_clip_val": trainer_params["gradient_clip_val"]
            }
        else:
            additional_params = {}
        if "min_lr" in trainer_params:
            trainer = Trainer(
                max_epochs=trainer_params["max_epochs"],
                logger=wandb_logger,
                enable_progress_bar=True,
                log_every_n_steps=15,
                default_root_dir=
                f"./models/checkpoints/{_config['db_collection']}-{_config['overwrite']}",
                detect_anomaly=True,
                callbacks=[
                    EarlyStopping(monitor="lr",
                                  mode="min",
                                  stopping_threshold=trainer_params["min_lr"],
                                  check_on_train_epoch_end=True,
                                  min_delta=-1)
                ],
                **additional_params)
        else:
            trainer = Trainer(
                max_epochs=trainer_params["max_epochs"],
                logger=wandb_logger,
                enable_progress_bar=True,
                log_every_n_steps=15,
                default_root_dir=
                f"./models/checkpoints/{_config['db_collection']}-{_config['overwrite']}",
                detect_anomaly=True,
                **additional_params)

        trainer.fit(self.lightning_model,
                    train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader)
        if trainer_params["testing"] == True:
            result = trainer.test(self.lightning_model, self.test_loader)
            wandb.finish()
            return result
        else:
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
