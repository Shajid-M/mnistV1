from typing import Callable, Iterator
import pytorch_lightning as pl
from torch import Tensor, optim
from torchmetrics import Accuracy
from torch import nn

PARTIAL_OPTIMIZER_TYPE = Callable[[Iterator[nn.Parameter]], optim.Optimizer]


class TrainingTask(pl.LightningModule):
    def __init__(self, optimizer: PARTIAL_OPTIMIZER_TYPE):
        super().__init__()
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
    

class MNISTClassification(TrainingTask):
    def __init__(self, model: nn.Module, optimizer: PARTIAL_OPTIMIZER_TYPE, loss_function: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss_function = loss_function

        classes = 10
        self.train_accuracy = Accuracy(task='multiclass', num_classes=classes)
        self.validation_accuracy = Accuracy(task='multiclass', num_classes=classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=classes)

    def forward(self, x) -> None:
        return self.model(x)
    
    def training_step(self, batch, _) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_function(logits, labels)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _) -> None:
        images, labels = batch
        preds = self(images)
        self.validation_accuracy(preds, labels)
        self.log("validation_accuracy", self.validation_accuracy, on_step=False, on_epoch=True)
    
    def tesing_step(self, batch, _) -> None:
        images, labels = batch
        preds = self(images)
        self.test_accuracy(preds, labels)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)