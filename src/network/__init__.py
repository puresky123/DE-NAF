from .network import DensityNetwork
from .deNAF import deNafVMSplit, deNafCP

def get_network(type):
    if type == "mlp": # NAF
        return DensityNetwork
    elif type == "tensor": # de_NAF
        return deNafVMSplit
    elif type == "tensor_cp": # de_NAF
        return deNafCP
    else:
        raise NotImplementedError("Unknown network type!")

