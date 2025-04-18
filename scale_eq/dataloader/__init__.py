from .mnist_data_module import MNISTDataModule, ScaleMNISTDataModule
from .stl_data_module import ScaleSTLDataModule
from .yesno_data_module import YESNODataModule
available_pl_modules = {
    'mnist': MNISTDataModule,
    'scale_mnist': ScaleMNISTDataModule,
    'scale_stl': ScaleSTLDataModule,
    'yesno': YESNODataModule
}


def get_pl_datamodule(name):
    return available_pl_modules[name]


def get_available_pl_modules():
    return list(available_pl_modules.keys())
