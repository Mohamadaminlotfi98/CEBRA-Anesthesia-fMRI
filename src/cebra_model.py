"""
Sklearn-compatible wrapper around CEBRA.

Implements the supervised contrastive representation-learning model
described in Methods ("Label-Guided Manifold Learning via CEBRA" and
"Network Architecture and Training Dynamics"): the offset10-model-mse
1D-CNN encoder trained with an InfoNCE loss to map ROI-activity windows
to a 3-D latent space, using discrete anesthesia-state labels to define
positive/negative pairs.
"""

from __future__ import annotations

import numpy as np
import torch
from cebra import CEBRA
from sklearn.base import BaseEstimator, TransformerMixin

from config import CEBRA_PARAMS


class CEBRAWrapper(BaseEstimator, TransformerMixin):
    """Wraps `cebra.CEBRA` so it can be used inside sklearn pipelines,
    GroupKFold cross-validation, and ituna's ConsistencyEnsemble.
    """

    def __init__(
        self,
        random_state=None,
        model_architecture=CEBRA_PARAMS["model_architecture"],
        output_dimension=CEBRA_PARAMS["output_dimension"],
        batch_size=CEBRA_PARAMS["batch_size"],
        learning_rate=CEBRA_PARAMS["learning_rate"],
        max_iterations=CEBRA_PARAMS["max_iterations"],
        conditional=CEBRA_PARAMS["conditional"],
        distance=CEBRA_PARAMS["distance"],
        num_hidden_units=CEBRA_PARAMS["num_hidden_units"],
        time_offsets=CEBRA_PARAMS["time_offsets"],
        temperature=CEBRA_PARAMS["temperature"],
        device=CEBRA_PARAMS["device"],
        verbose=CEBRA_PARAMS["verbose"],
    ):
        self.random_state = random_state
        self.model_architecture = model_architecture
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.conditional = conditional
        self.distance = distance
        self.num_hidden_units = num_hidden_units
        self.time_offsets = time_offsets
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        self.model_ = CEBRA(
            model_architecture=self.model_architecture,
            output_dimension=self.output_dimension,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
            conditional=self.conditional,
            distance=self.distance,
            num_hidden_units=self.num_hidden_units,
            time_offsets=self.time_offsets,
            temperature=self.temperature,
            device=self.device,
            verbose=self.verbose,
        )
        self.model_.fit(X, y.astype(int))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model_.transform(X)
