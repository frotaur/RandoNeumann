import torch

def ord_string_state(state_num):
    """
        Given the state, returns a (5,) tensor for the arrows, and (5,) for the excitations
        
    """
    if(state_num==1):
        return torch.tensor([1,1,1,1,1]),torch.tensor([0,0,0,0,1])