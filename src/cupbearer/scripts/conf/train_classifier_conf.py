from dataclasses import dataclass, field

from cupbearer.data import (
    BackdoorData,
    DatasetConfig,
    TamperingDataConfig,
    WanetBackdoor,
)
from cupbearer.models import CNNConfig, MLPConfig, ModelConfig, TamperTransformerConfig
from cupbearer.utils.scripts import ScriptConfig
from cupbearer.utils.train import DebugTrainConfig, TrainConfig

from .._shared import ClassificationTask


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: ModelConfig
    train_config: TrainConfig = field(default_factory=TrainConfig)
    train_data: DatasetConfig
    val_data: dict[str, DatasetConfig] = field(default_factory=dict)
    task: ClassificationTask = "multiclass"
    # If True, returns the Lighting Trainer object (which has the model and a bunch
    # of other information, this may be useful when using interactively).
    # Otherwise (default), return only a dictionary of latest metrics, to avoid e.g.
    # submitit trying to pickle the entire Trainer object.
    return_trainer: bool = False

    @property
    def num_classes(self):
        return self.train_data.num_classes

    @property
    def num_labels(self):
        return self.train_data.num_labels

    def __post_init__(self):
        super().__post_init__()
        # HACK: Need to add new architectures here as they get implemented.
        if isinstance(self.model, (MLPConfig, CNNConfig)):
            self.model.output_dim = self.num_classes

        if isinstance(self.model, TamperTransformerConfig):
            assert isinstance(self.train_data, TamperingDataConfig)
            self.model.n_sensors = self.train_data.n_sensors

        # For datasets that are not necessarily deterministic based only on
        # arguments, this is where validation sets are set to follow train_data
        if isinstance(self.train_data, BackdoorData):
            for name, val_config in self.val_data.items():
                # WanetBackdoor
                if (
                    isinstance(self.train_data.backdoor, WanetBackdoor)
                    and isinstance(val_config, BackdoorData)
                    and isinstance(val_config.backdoor, WanetBackdoor)
                ):
                    str_factor = (
                        val_config.backdoor.warping_strength
                        / self.train_data.backdoor.warping_strength
                    )
                    val_config.backdoor.control_grid = (
                        str_factor * self.train_data.backdoor.control_grid
                    )


@dataclass
class DebugConfig(Config):
    train_config: DebugTrainConfig = field(default_factory=DebugTrainConfig)
