from sacred import Experiment
import numpy as np
import seml
from typing import List, Dict, Optional
import datasets.data
from models.gcn import GCN
from models.substructure_model import SubstructureNeuralNet
from models.lightning_models import *

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
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
    def init_dataset(self, _config, dataset: str, substructures: Dict[str, List[int]], target: Optional[str] = None, remove_node_features: bool = False, loader_params: Optional[Dict] = None):
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
        loader_params, optional
            Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.
        """
        print(f"Dataset received config: {config}")
        self.train_loader, self.val_loader, self.test_loader, self.num_features, self.num_classes, self.num_substructures  = datasets.data.load(dataset, substructures, target, remove_node_features, loader_params)


    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_params: dict, classification: bool = True):
        self.classification = classification
        model_params = model_params.copy() # allows us to add fields to it
        model_params["out_channels"] = self.num_classes if classification else 1
        model_params["in_channels"] = self.num_features
        if model_type == "GCN":
            self.model = GCN(**model_params)
        elif model_type == "SubstrucutureNet":
            model_params["num_substructures"] = self.num_substructures
            self.model = SubstructureNeuralNet(**model_params)
        print("Setup model:")
        print(self.model)

    @ex.capture(prefix="optimization")
    def init_optimizer(self, optimization_params):
        if self.classification:
            loss = classification_loss
            acc = classification_accuracy
        else:
            loss = regression_loss
            acc = regression_acc

        self.lightning_model = LightningModel(model = self.model, loss = loss, acc = acc, optimizer_parameters= optimization_params)

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

        trainer = Trainer(max_epochs=trainer_params["max_epochs"], logger= wandb_logger, enable_progress_bar= False, log_every_n_steps=15)

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