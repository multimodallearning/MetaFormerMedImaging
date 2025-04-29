import timm
from clearml import Task

from models.med_mnist_base import MedMNISTBase


class CNNClassifier(MedMNISTBase):
    def __init__(self, dataset_name: str, model: str = 'resnet34', pretrained: bool = False):
        super().__init__(dataset_name)
        self.model_name = model
        self.ds_name = dataset_name
        if self.is_2d:
            self.model = timm.create_model(model, pretrained=pretrained, num_classes=self.n_classes, in_chans=self.n_channels)
        elif self.is_3d:
            raise NotImplementedError("3D models not implemented yet")

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        if Task.current_task() is not None:
            Task.current_task().set_name(f'{self.model_name}_{self.ds_name}')
            Task.current_task().set_tags([f'{self.ds_name}', f'{self.model_name}'])
