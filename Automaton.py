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

        self.state = torch.roll(self.state,random.randint(0,3),dims=1)
        self.state = torch.roll(self.state,random.randint(0,3),dims=2)
        # self.state = torch.roll(self.state,random.randint(0,3),dims=0)
        self.state[:,self.h//2,self.w//2]=torch.rand(3, device=self.device)
        


    def draw(self):
        """
            Generates the worldmap from the state of the automaton.
        """

        self._worldmap= self.state


class VonNeumann(Automaton):
    """
        Von Neumann cellular automaton

        Parameters :
        size : (H,W)
    """
    
    def __init__(self, size, device='cpu'):
        super().__init__(size, device=device)

        state = torch.randint(1,10,(1,*size), device=device)

        # state = torch.zeros_like(state)

        # Excitations :
        self.excitations = (torch.rand((1,*size), device=device)<0.8).to(dtype=torch.uint8)

        # self.excitations[0,self.h//2,self.w//2]=1
        # Ordinary transmissions :
        self.ord_e = torch.where(state==1,1,0).to(torch.uint8)
        self.ord_w = torch.where(state==2,1,0).to(torch.uint8)
        self.ord_s = torch.where(state==3,1,0).to(torch.uint8)
        self.ord_n = torch.where(state==4,1,0).to(torch.uint8)
        self.compute_is_ord()
        
        # Special transmission :
        self.spe_e = torch.where(state==5,1,0).to(torch.uint8)
        self.spe_w = torch.where(state==6,1,0).to(torch.uint8)
        self.spe_s = torch.where(state==7,1,0).to(torch.uint8)
        self.spe_n = torch.where(state==8,1,0).to(torch.uint8)
        self.compute_is_spe()


        # Confluent :
        self.is_conf = torch.where(state==9,1,0).to(torch.uint8)
        self.conf_in = torch.zeros_like(self.is_conf)
        self.conf_out = torch.zeros_like(self.is_conf)

        # Sensitized :
        self.is_sens = torch.where(state==10,1,0).to(torch.uint8)
        self.sens_state = torch.where(self.is_sens,1,0 )
        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # Init self.inc_ords and self.inc_spes
        self.compute_ord_excitations()
        self.compute_spe_excitations()
        self.compute_is_ord()
        self.compute_is_spe()
        self.compute_is_ground()

        self._worldmap = torch.zeros((1,self.h,self.w), device=device, dtype=torch.int)

    def compute_is_ord(self):
        self.is_ord = ((self.ord_e+self.ord_w+self.ord_s+self.ord_n)>0).to(torch.uint8)
    
    def compute_is_spe(self):
        self.is_spe = ((self.spe_e+self.spe_w+self.spe_s+self.spe_n)>0).to(torch.uint8)
    
    def compute_is_killed(self):
        self.compute_is_ord()
        self.compute_is_spe()

        self.is_killed = ((self.is_ord*self.inc_spes + self.is_spe*self.inc_ords + self.is_conf*self.inc_spes)>0).to(torch.uint8)

    def compute_is_ground(self):
        self.is_ground = ((1-self.is_ord)*(1-self.is_spe)*(1-self.is_conf)).to(torch.uint8)


    def compute_conf(self):
        self.compute_inhibitions()

        inc_conf_e = torch.roll(self.conf_out*self.is_conf,shifts=-1,dims=2)
        inc_conf_w = torch.roll(self.conf_out*self.is_conf,shifts=+1,dims=2)
        inc_conf_s = torch.roll(self.conf_out*self.is_conf,shifts=-1,dims=1)
        inc_conf_n = torch.roll(self.conf_out*self.is_conf,shifts=+1,dims=1)

        self.inc_conf_ords = self.is_ord*(inc_conf_e*(1-self.ord_e)+inc_conf_w*(1-self.ord_w)+inc_conf_s*(1-self.ord_s)+inc_conf_n*(1-self.ord_n))
        self.inc_conf_spes = self.is_spe*(inc_conf_e*(1-self.spe_e)+inc_conf_w*(1-self.spe_w)+inc_conf_s*(1-self.spe_s)+inc_conf_n*(1-self.spe_n))

        self.conf_out = self.conf_in*self.is_conf
        self.conf_in = (1-(self.inh))*self.inc_ords*self.is_conf

    def compute_sens(self):
        inc_exc = ((self.inc_ords+self.inc_spes)>0).to(torch.uint8)
        
        self.is_sens = torch.where((self.is_ground)*inc_exc,1,self.is_sens)

        self.sens_state = ((self.sens_state << 1)+inc_exc)*self.is_sens

        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)
        self.births = torch.where((self.sens_state==0b10000),1,self.births) # East
        self.births = torch.where((self.sens_state==0b10001),4,self.births) # North
        self.births = torch.where((self.sens_state==0b1001),2,self.births) # West
        self.births = torch.where((self.sens_state==0b1010),3,self.births) # South

        self.births = torch.where((self.sens_state==0b1011),1+4,self.births) # East Special
        self.births = torch.where((self.sens_state==0b1100),4+4,self.births) # North Special
        self.births = torch.where((self.sens_state==0b1101),2+4,self.births) # West Special
        self.births = torch.where((self.sens_state==0b1110),3+4,self.births) # South Special

        self.births = torch.where((self.sens_state==0b1111),9,self.births) # Confluent

        self.is_sens = torch.where(self.births>0,0,self.is_sens)
    
    def make_births(self):
        self.ord_e = torch.where(self.births==1,1,self.ord_e)
        self.ord_n = torch.where(self.births==4,1,self.ord_n)
        self.ord_w = torch.where(self.births==2,1,self.ord_w)
        self.ord_s = torch.where(self.births==3,1,self.ord_s)

        self.spe_e = torch.where(self.births==5,1,self.spe_e)
        self.spe_n = torch.where(self.births==8,1,self.spe_n)
        self.spe_w = torch.where(self.births==6,1,self.spe_w)
        self.spe_s = torch.where(self.births==7,1,self.spe_s)

        self.is_conf = torch.where(self.births==9,1,self.is_conf)

    def compute_ord_excitations(self):
        inc_ord_e = torch.roll(self.ord_w*self.excitations,shifts=-1,dims=2)
        inc_ord_w = torch.roll(self.ord_e*self.excitations,shifts=1,dims=2)
        inc_ord_s = torch.roll(self.ord_n*self.excitations,shifts=-1,dims=1)
        inc_ord_n = torch.roll(self.ord_s*self.excitations,shifts=1,dims=1)

        self.inc_ords = inc_ord_e*(1-self.ord_e)+inc_ord_w*(1-self.ord_w)+inc_ord_s*(1-self.ord_s)+inc_ord_n*(1-self.ord_n)

    def compute_inhibitions(self):
        inc_inh_e = torch.roll(self.ord_w*(1-self.excitations),shifts=-1,dims=2)
        inc_inh_w = torch.roll(self.ord_e*(1-self.excitations),shifts=1,dims=2)
        inc_inh_s = torch.roll(self.ord_n*(1-self.excitations),shifts=-1,dims=1)
        inc_inh_n = torch.roll(self.ord_s*(1-self.excitations),shifts=1,dims=1)

        self.inh = ((inc_inh_e+inc_inh_w+inc_inh_s+inc_inh_n)>0).to(torch.uint8)

    def compute_spe_excitations(self):
        """
            Computes the excitations of the special particles, and returns the purged incoming excitations.
        """
        inc_spe_e = torch.roll(self.spe_w*self.excitations,shifts=-1,dims=2)
        inc_spe_w = torch.roll(self.spe_e*self.excitations,shifts=1,dims=2)
        inc_spe_s = torch.roll(self.spe_n*self.excitations,shifts=-1,dims=1)
        inc_spe_n = torch.roll(self.spe_s*self.excitations,shifts=1,dims=1)

        self.inc_spes = inc_spe_e*(1-self.spe_e)+inc_spe_w*(1-self.spe_w)+inc_spe_s*(1-self.spe_s)+inc_spe_n*(1-self.spe_n)
    
    def step(self):
        self.compute_ord_excitations()
        self.compute_spe_excitations()

        self.compute_conf()
        self.compute_sens()

        


        self.excitations = ((self.inc_ords+self.inc_spes+self.inc_conf_ords+self.inc_conf_spes)>0).to(torch.int)
        # self.excitations = ((self.inc_ords+self.inc_spes)>0).to(torch.int)

        self.compute_is_killed()
        self.kill_dead()
        
        self.make_births()
        self.compute_is_ground()

        
    def kill_dead(self):
        is_alive = (1-self.is_killed)
        self.ord_e = self.ord_e*is_alive
        self.ord_w = self.ord_w*is_alive
        self.ord_s = self.ord_s*is_alive
        self.ord_n = self.ord_n*is_alive

        self.spe_e = self.spe_e*is_alive
        self.spe_w = self.spe_w*is_alive
        self.spe_s = self.spe_s*is_alive
        self.spe_n = self.spe_n*is_alive

        self.is_conf = self.is_conf*is_alive
        self.conf_in = self.conf_in*is_alive
        self.conf_out = self.conf_out*is_alive

        self.compute_is_ord()
        self.compute_is_spe()

        self.excitations = self.excitations*is_alive

    def draw(self):
        self._worldmap = torch.zeros_like(self._worldmap)
        self._worldmap = torch.where(self.ord_e==1,1,self._worldmap)
        self._worldmap = torch.where(self.ord_w==1,2,self._worldmap)
        self._worldmap = torch.where(self.ord_s==1,3,self._worldmap)
        self._worldmap = torch.where(self.ord_n==1,4,self._worldmap)
        self._worldmap = torch.where(self.spe_e==1,5,self._worldmap)
        self._worldmap = torch.where(self.spe_w==1,6,self._worldmap)
        self._worldmap = torch.where(self.spe_s==1,7,self._worldmap)
        self._worldmap = torch.where(self.spe_n==1,8,self._worldmap)
        self._worldmap = torch.where(self.is_conf==1,9,self._worldmap)
        self._worldmap = torch.where(self.is_sens==1,10,self._worldmap)
    
    @property
    def worldmap(self):
        # OVERRIDE WORLDMAP TO CONTAIN TYPES OF PARTICLES
        return self._worldmap.permute(2,1,0).cpu().numpy(), self.excitations.permute(2,1,0).cpu().numpy(), self.conf_out.permute(2,1,0).cpu().numpy()