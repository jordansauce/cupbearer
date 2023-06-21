from abc import ABC, abstractmethod
from loguru import logger
from matplotlib import pyplot as plt
import sklearn.metrics

from torch.utils.data import DataLoader

import flax.linen as nn
import jax
import jax.numpy as jnp

from abstractions import data


class AnomalyDetector(ABC):
    def __init__(self, model: nn.Module, params, max_batch_size: int = 4096):
        self.model = model
        self.params = params
        self.max_batch_size = max_batch_size

        self.forward_fn = jax.jit(
            lambda x: model.apply({"params": params}, x, return_activations=True)
        )

        self.trained = False

    def _model(self, batch):
        # batch may contain labels or other info, if so we strip it out
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        output, activations = self.forward_fn(inputs)
        return output, activations

    def train(self, *args, **kwargs):
        """Train the anomaly detector with the given dataset as "normal" data."""
        self.trained = True
        return self._train(*args, **kwargs)

    def scores(self, batch):
        """Compute anomaly scores for the given inputs.

        Args:
            inputs: a batch of input data to the model (potentially including labels).

        Returns:
            A batch of anomaly scores for the inputs.
        """
        if not self.trained:
            raise RuntimeError("Anomaly detector must be trained first.")
        return self._scores(batch)

    def eval(self, normal_dataset, anomalous_dataset):
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )
        anomalous_loader = DataLoader(
            anomalous_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )

        normal_scores = []
        for batch in normal_loader:
            normal_scores.append(self.scores(batch))
        normal_scores = jnp.concatenate(normal_scores)

        anomalous_scores = []
        for batch in anomalous_loader:
            anomalous_scores.append(self.scores(batch))
        anomalous_scores = jnp.concatenate(anomalous_scores)

        true_labels = jnp.concatenate(
            [jnp.ones_like(anomalous_scores), jnp.zeros_like(normal_scores)]
        )
        auc_roc = sklearn.metrics.roc_auc_score(
            y_true=true_labels,
            y_score=jnp.concatenate([anomalous_scores, normal_scores]),
        )
        logger.log("METRICS", f"AUC_ROC: {auc_roc:.4f}")

        # Visualizations for consistency losses
        plt.hist(normal_scores, bins=100, alpha=0.5, label="Normal")
        plt.hist(
            anomalous_scores,
            bins=100,
            alpha=0.5,
            label="Anomalous",
        )
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.title("Anomaly score distribution")
        plt.savefig("histogram.pdf")

    @abstractmethod
    def _train(self, dataset):
        pass

    @abstractmethod
    def _scores(self, batch):
        pass
