import numpy as np
import random, torch
from torchenhanced import DevModule

class Automaton(DevModule) :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pytorch, the world tensor has shape
        (3,H,W). It contains float values between 0 and 1, which
        are (automatically) mapped to 0 255 when returning output, 
        and describes how the world is 'seen' by an observer.

        Parameters :
        size : 2-uple (H,W)
            Shape of the CA world
        
        Attributes :
        worldmap : (W,H,3) np.ndarray of uints between 0 and 255, ready for plotting in pygame
        
    """

    

    def __init__(self,size, device='cpu'):
        super().__init__()
        self.h, self.w  = size
        self.size = size
        # This self._worldmap should be changed in the step function.
        # It should contains floats from 0 to 1 of RGB values.
        # Has shape (3,H,W) as the pytorch conventions.

        # <<<FOR GPU AND EASY DEVICE SWITCHING, SWITCH THIS TO REGISTER_BUFFER>>>
        self._worldmap = torch.rand((3,self.h,self.w), device=device)
    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap.permute(2,1,0).cpu().numpy()).astype(dtype=np.uint8)

        
class EmptyAuto(Automaton):
    """
        Particle based simulation

        Parameters :
        size : (H,W)

    """

    def __init__(self, size , device='cpu'):
        super().__init__(size, device=device)

        self.state = torch.zeros((3,self.h,self.w), device=device)
        self.state[0,self.h//2,self.w//2]=1
        
        
                    
    def step(self):
        """
            Steps the automaton state by one iteration.
        """
        bef = self.state[:,self.h//2,self.w//2]
        self.state = torch.roll(self.state,random.randint(0,3),dims=1)
        self.state = torch.roll(self.state,random.randint(0,3),dims=2)
        # self.state = torch.roll(self.state,random.randint(0,3),dims=0)
        self.state[:,self.h//2,self.w//2]=torch.rand(3, device=self.device)
        
        self.draw()
    

    def draw(self):
        """
            Generates the worldmap from the state of the automaton.
        """

        self._worldmap= self.state


