import torch

def ord_string_state(state_num):
    """
        Given the state, returns a (5,) tensor for the arrows, and (5,) for the excitations
    """
    arrows = torch.tensor([1,1,1,1,1])
    if(state_num==1):
        return arrows,torch.tensor([0,0,0,0,1])
    elif(state_num==2):
        return arrows,torch.tensor([0,1,0,0,1])
    elif(state_num==3):
        return arrows,torch.tensor([0,0,1,0,1])
    elif(state_num==4):
        return arrows,torch.tensor([1,0,0,0,1])
    elif(state_num==5):
        return arrows,torch.tensor([0,1,1,0,1])
    elif(state_num==6):
        return arrows,torch.tensor([0,1,0,1,1])
    elif(state_num==7):
        return arrows,torch.tensor([0,0,1,1,1])
    elif(state_num==8):
        return arrows,torch.tensor([0,0,0,1,1])
    elif(state_num==9):
        return arrows,torch.tensor([0,1,1,1,1])

def get_sens_benchmark():
    """
        Return a portion of state which is the benchmark for the sensitive cells,
        it is of size (5,9)*2 (arrows, excitations)
    """
    out_state = torch.zeros((9,5))
    out_excit = torch.zeros((9,5))
    for i in range(1,10):
        out_state[i-1,:],out_excit[i-1,:] = ord_string_state(i)
    
    return out_state,out_excit