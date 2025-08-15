from konfai.network import network
import segmentation_models_pytorch as smp
import torch
from konfai.utils.config import config

class Head(network.ModuleArgsDict):

    def __init__(self):
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())

class UNetpp(network.Network):
    
    def __init__(self,
                optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                schedulers: dict[str, network.LRSchedulersLoader] = {
                    "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
                },
                outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                pretrained: bool = False):
        super().__init__(in_channels = 3, optimizer = optimizer, schedulers = schedulers, outputs_criterions = outputs_criterions, dim = 2)
        self.add_module("model", smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None if not pretrained else "imagenet", 
            in_channels=3,
            classes=1,  
            activation=None  
        ))
        self.add_module("Head", Head())