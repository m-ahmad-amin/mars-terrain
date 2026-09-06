from dataclasses import dataclass

NUM_CLASSES = 4
IGNORE_INDEX = 255
CLASS_NAMES = ("soil", "bedrock", "sand", "big rock")


@dataclass
class TrainConfig:
    image_size: int = 512
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    max_train: int | None = None
    seed: int = 0
    amp: bool = True
    resume: bool = True
