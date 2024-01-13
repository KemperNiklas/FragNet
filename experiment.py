from sacred import Experiment
import numpy as np
import seml
from typing import List, Dict, Optional
import datasets.data
from models.gcn import GCN, VerySimpleGCN
from models.substructure_model import SubstructureNeuralNet
from models.pool_linear import PoolLinear
from models.lightning_models import *

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import wandb

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
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class ExperimentWrapper:

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, _config, dataset: str, seed: Optional[int] = None, substructures: Dict[str, List[int]] = {}, target = [], remove_node_features: bool = False, one_hot_degree: bool = False, substructure_node_feature: List = [], loader_params: Optional[Dict] = None):
        """Initialize train, validation and test loader.

        Parameters
        ----------
        dataset
            Name of the dataset
        substructures
            Substructures to be considered in message passing and their sizes.
            Supported substrucutres are ``ring`` and ``clique``.
        target, optional
            Replace target by count of a motif with given size (either node_level or not). target = (motif, size, node_level).
        remove_node_features, optional
            Boolean indicating whether node_labels should be removed, by default False
        one_hot_degree, optional
            Boolean indicating whether to concatinate the node features with a one hot encoded node degree, by default False.
        loader_params, optional
            Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.
        """
        print(f"Dataset received config: {_config}")
        if seed:
            torch.manual_seed(seed)
        self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes, self.num_substructures  = datasets.data.load(dataset, substructures, target, remove_node_features, one_hot_degree, substructure_node_feature, loader_params)


    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_params: dict, classification: bool = True):
        self.classification = classification
        model_params = model_params.copy() # allows us to add fields to it
        model_params["out_channels"] = self.num_classes if classification and self.num_classes > 2 else 1
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
        else:
            raise RuntimeError(f"Model {model_type} not supported")
        print("Setup model:")
        print(self.model)

    @ex.capture(prefix="optimization")
    def init_optimizer(self, optimization_params, scheduler_parameters = None, loss: Optional[str] = None, additional_metric: Optional[str] = None):
        if self.classification and self.num_classes > 2:
            loss_fun = ce_loss
            acc = classification_accuracy
        elif self.classification:
            loss_fun = bce_loss
            acc = binary_classification_accuracy
        else:
            if loss and loss == "mae":
                loss_func = mae_loss
            else:
                loss_func = mse_loss
            acc = regression_acc
        
        if additional_metric:
            if additional_metric == "mae":
                additional_metric_func = mae_loss
            elif additional_metric == "mse":
                additional_metric_func = mse_loss

        self.lightning_model = LightningModel(model = self.model, loss = loss_func, acc = acc, optimizer_parameters= optimization_params,scheduler_parameters=scheduler_parameters, additional_metric= additional_metric_func)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    @ex.capture()
    def train(self, trainer_params, project_name, _config):

        wandb_logger = WandbLogger(project=project_name)
        wandb_logger.experiment.config.update(_config)
        #wandb_logger.watch(self.lightning_model, log="all")

        if "min_lr" in trainer_params:
            trainer = Trainer(max_epochs=trainer_params["max_epochs"], logger= wandb_logger, enable_progress_bar= True, log_every_n_steps=15, default_root_dir=f"./models/checkpoints/{_config['db_collection']}-{_config['overwrite']}", detect_anomaly= True, 
                              callbacks = [EarlyStopping(monitor = "lr", mode = "min", stopping_threshold = trainer_params["min_lr"], check_on_train_epoch_end=True, min_delta=-1)])
        else:
            trainer = Trainer(max_epochs=trainer_params["max_epochs"], logger= wandb_logger, enable_progress_bar= True, log_every_n_steps=15, default_root_dir=f"./models/checkpoints/{_config['db_collection']}-{_config['overwrite']}", detect_anomaly= True)
            
        trainer.fit(self.lightning_model, train_dataloaders=self.train_loader, val_dataloaders= self.val_loader)
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