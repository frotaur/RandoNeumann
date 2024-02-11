import numpy as np
import random, torch
from torchenhanced import DevModule
from init_maker import get_sens_benchmark


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

        self.to(device)

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

        self.p_ordinary, self.p_special, self.p_confluent = 19.,.0,.0
        state = self.get_rando_para_state(self.p_ordinary, self.p_special, self.p_confluent)

        
        # Excitations :
        self.excitations = (torch.rand((1,*size), device=device)<0.2).to(dtype=torch.uint8)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state)

        # self.run_mcmc(100, 3, replace_prob=.2)
        self._worldmap = torch.zeros((1,self.h,self.w), device=device, dtype=torch.int)

        self.to(device)
        self.p_ind = (0, 53, 70)
    def inj_excitations(self):
        self.excitations = (torch.rand((1,*self.size), device=self.device)<0.2).to(dtype=torch.uint8)

    def get_rando_para_state(self, p_ordinary=None, p_special=None, p_confluent=None, p_dead=0.,
                             state_size=None, batch_size=1):
        """
            Returns a random batch of states with probabilities given as parameters.

            Parameters :
            p_ordinary : Probability of ordinary transmission, if None, uses self
            p_special : Probability of special transmission, if None, uses self
            p_confluent : Probability of confluent, if None, uses self
            p_dead : Probability of dead cell
            state_size : size of the state. If None, uses self.size
            batch_size : Number of states to generate

            Return :
            state : (B,H,W) tensor of uint8
        """
        p_ordinary = self.p_ordinary if p_ordinary is None else p_ordinary
        p_special = self.p_special if p_special is None else p_special
        p_confluent = self.p_confluent if p_confluent is None else p_confluent
        state_size = self.size if state_size is None else state_size

        p_total = p_ordinary+p_special+p_confluent+p_dead

        p_ordinary /= p_total
        p_special /= p_total
        p_confluent /= p_total
        p_dead /= p_total

        state = torch.zeros((batch_size,*state_size)).to(self.device)

        # States random :
        uniform = torch.rand(state.shape).to(self.device)

        ordinary_locs = uniform < p_ordinary
        special_locs = (p_ordinary < uniform) & (uniform < p_ordinary + p_special)
        confluent_locs = (p_ordinary + p_special < uniform) & (uniform < p_ordinary + p_special + p_confluent)

        transmission_dirs = torch.randint_like(state, 0, 4)
        state = torch.where(ordinary_locs, transmission_dirs + 1, 0)
        state += torch.where(special_locs, transmission_dirs + 5, 0)
        state += torch.where(confluent_locs, 9, 0)

        return state

    def set_state(self, state, excitations = None):
        """
        	Fills all the tensor for each species given by the state, which should be a tensor of
            shape (1,H,W), with values between 0 and 9. If provided, sets excitations to the given
            (1,H,W) tensor of boolean/int(0,1) values. Otherwise, 0 excitations are set.
            TODO : provide option for random excitations.

            Args :
            state : (B,H,W) tensor of uint8
            excitations : (B,H,W) tensor of uint8 or booleans
        """
        if(excitations is None):
            excitations = torch.zeros_like(state, dtype=torch.uint8).to(self.device)
        else :
            excitations = (excitations>0).to(self.device, dtype=torch.uint8)
        assert state.shape == excitations.shape, "State and excitations should have the same shape"

        state = state.to(self.device)
        # Ordinary transmissions :
        self.ord_e = torch.where(state==1,1,0).to(torch.uint8)
        self.ord_w = torch.where(state==2,1,0).to(torch.uint8)
        self.ord_s = torch.where(state==3,1,0).to(torch.uint8)
        self.ord_n = torch.where(state==4,1,0).to(torch.uint8)
        
        # Special transmission :
        self.spe_e = torch.where(state==5,1,0).to(torch.uint8)
        self.spe_w = torch.where(state==6,1,0).to(torch.uint8)
        self.spe_s = torch.where(state==7,1,0).to(torch.uint8)
        self.spe_n = torch.where(state==8,1,0).to(torch.uint8)

        # Confluent :
        self.is_conf = torch.where(state==9,1,0).to(torch.uint8)
        self.conf_in = torch.zeros_like(self.is_conf)
        self.conf_out = torch.zeros_like(self.is_conf)

        # Sensitized :
        self.is_sens = torch.where(state==10,1,0).to(torch.uint8)
        self.sens_state = torch.where(self.is_sens,1,0 )
        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # Killed :
        self.is_killed = torch.zeros_like(state,dtype=torch.uint8)

        # Init self.inc_ords and self.inc_spes
        self.compute_is_ord()
        self.compute_is_spe()
        # self.compute_ord_excitations()
        # self.compute_spe_excitations()
        self.compute_is_ground()
        
        self.excitations = excitations.to(self.device)



    def get_state(self):
        """
            Returns a tensor of shape (1,H,W), with values between 0 and 9.
        """
        state = torch.zeros((1,*self.size), device=self.device, dtype=torch.int)

        # Ordinary transmissions :
        state = torch.where(self.ord_e==1,1,state)
        state = torch.where(self.ord_w==1,2,state)
        state = torch.where(self.ord_s==1,3,state)
        state = torch.where(self.ord_n==1,4,state)
        
        # Special transmission :
        state = torch.where(self.spe_e==1,5,state)
        state = torch.where(self.spe_w==1,6,state)
        state = torch.where(self.spe_s==1,7,state)
        state = torch.where(self.spe_n==1,8,state)

        # Confluent :
        state = torch.where(self.is_conf==1,9,state)
        
        # TODO : ALLOW SAVING ALL THE DIFFERENT SENSITIZED STATES
        # Sensitized :
        state = torch.where(self.is_sens==1,10,state)

        return state

    def reset_state(self):
        state = torch.randint(0,10,(1,*self.size), device=self.device)

        mask = torch.zeros_like(state)
        mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]=0
        # mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]=torch.randint(6,8,(1,70,70), device=self.device)
    
        state[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35] = mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]
        
        # Excitations :
        self.excitations = (torch.rand((1,*self.size), device=self.device)<0.5).to(dtype=torch.uint8)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state)
    
    def make_state_bench(self):
        """
            Replaces the state with the benchmark state
        """
        state= torch.zeros((1,*self.size), device=self.device, dtype=torch.int)
        self.excitations = torch.zeros((1,*self.size), device=self.device, dtype=torch.int)

        bench_state, bench_excit = get_sens_benchmark(True)
        state[0,2:9+2,2:5+2] = bench_state.to(self.device)
        self.excitations[0,2:9+2,2:5+2] = bench_excit.to(self.device)

        return state
    
    def compute_is_ord(self):
        self.is_ord = ((self.ord_e+self.ord_w+self.ord_s+self.ord_n)>0).to(torch.uint8)
    
    def compute_is_spe(self):
        self.is_spe = ((self.spe_e+self.spe_w+self.spe_s+self.spe_n)>0).to(torch.uint8)
    

    def compute_is_killed(self):
        self.compute_is_ord()
        self.compute_is_spe()

        self.is_killed = ((self.is_ord*self.inc_spes + self.is_spe*self.inc_ords + self.is_conf*self.inc_spes)>0).to(torch.uint8)

    def compute_is_ground(self):
        self.is_ground = ((1-self.is_ord)*(1-self.is_spe)*(1-self.is_conf)*(1-self.is_sens)).to(torch.uint8)


    def compute_conf(self):
        self.compute_inhibitions()

        conf_act = self.conf_out*self.is_conf
        inc_conf_e = torch.roll(conf_act,shifts=-1,dims=2)
        inc_conf_w = torch.roll(conf_act,shifts=+1,dims=2)
        inc_conf_s = torch.roll(conf_act,shifts=-1,dims=1)
        inc_conf_n = torch.roll(conf_act,shifts=+1,dims=1)
        
        # This looses an excitation on top of a confluent if put manually, but it's acceptable
        self.inc_conf_ords = self.is_ord*(inc_conf_e*(1-self.ord_e)+inc_conf_w*(1-self.ord_w)+inc_conf_s*(1-self.ord_s)+inc_conf_n*(1-self.ord_n))
        self.inc_conf_spes = self.is_spe*(inc_conf_e*(1-self.spe_e)+inc_conf_w*(1-self.spe_w)+inc_conf_s*(1-self.spe_s)+inc_conf_n*(1-self.spe_n))

        self.conf_out = self.conf_in*self.is_conf
        self.conf_in = (1-self.inh)*self.inc_ords*self.is_conf

    
    def compute_sens(self):
        inc_exc = ((self.inc_ords+self.inc_spes)>0).to(torch.uint8)

        self.is_sens = torch.where((self.is_ground)*inc_exc,1,self.is_sens)

        self.sens_state = ((self.sens_state << 1)+inc_exc)*self.is_sens

        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # I think this is very inefficient also, not sure how to batch it
        self.births = torch.where((self.sens_state==0b10000),1,self.births) # East
        self.births = torch.where((self.sens_state==0b1001),2,self.births) # West
        self.births = torch.where((self.sens_state==0b1010),3,self.births) # South
        self.births = torch.where((self.sens_state==0b10001),4,self.births) # North

        self.births = torch.where((self.sens_state==0b1011),1+4,self.births) # East Special
        self.births = torch.where((self.sens_state==0b1101),2+4,self.births) # West Special
        self.births = torch.where((self.sens_state==0b1110),3+4,self.births) # South Special
        self.births = torch.where((self.sens_state==0b1100),4+4,self.births) # North Special

        self.births = torch.where((self.sens_state==0b1111),9,self.births) # Confluent

        self.is_sens = torch.where(self.births>0,0,self.is_sens)
        self.sens_state = self.sens_state*self.is_sens
    
    def make_births(self):
        # Extinguish remaining excitations on birthing cells
        self.excitations = torch.where(self.births>0,0,self.excitations) 

        # This cannot be batched I think
        self.ord_e = torch.where(self.births==1,1,self.ord_e)
        self.ord_n = torch.where(self.births==4,1,self.ord_n)
        self.ord_w = torch.where(self.births==2,1,self.ord_w)
        self.ord_s = torch.where(self.births==3,1,self.ord_s)

        self.spe_e = torch.where(self.births==5,1,self.spe_e)
        self.spe_n = torch.where(self.births==8,1,self.spe_n)
        self.spe_w = torch.where(self.births==6,1,self.spe_w)
        self.spe_s = torch.where(self.births==7,1,self.spe_s)

        self.is_conf = torch.where(self.births==9,1,self.is_conf)

        self.births = torch.zeros_like(self.births)
        # Recompute which the 'is' thingies
        self.compute_is_ord()
        self.compute_is_spe()

    def compute_ord_excitations(self):
        # This cannot be batched I think
        inc_ord_e = torch.roll(self.ord_w*self.excitations,shifts=-1,dims=2)
        inc_ord_w = torch.roll(self.ord_e*self.excitations,shifts=1,dims=2)
        inc_ord_s = torch.roll(self.ord_n*self.excitations,shifts=-1,dims=1)
        inc_ord_n = torch.roll(self.ord_s*self.excitations,shifts=1,dims=1)

        self.inc_ords = inc_ord_e*(1-self.ord_e)+inc_ord_w*(1-self.ord_w)+inc_ord_s*(1-self.ord_s)+inc_ord_n*(1-self.ord_n)

    def compute_inhibitions(self):
        # This cannot be batched I think
        inc_inh_e = torch.roll(self.ord_w*(1-self.excitations),shifts=-1,dims=2)
        inc_inh_w = torch.roll(self.ord_e*(1-self.excitations),shifts=1,dims=2)
        inc_inh_s = torch.roll(self.ord_n*(1-self.excitations),shifts=-1,dims=1)
        inc_inh_n = torch.roll(self.ord_s*(1-self.excitations),shifts=1,dims=1)

        self.inh = ((inc_inh_e+inc_inh_w+inc_inh_s+inc_inh_n)>0).to(torch.uint8)

    def compute_spe_excitations(self):
        """
            Computes the excitations of the special particles, and returns the purged incoming excitations.
        """
        # This cannot be batched I think
        inc_spe_e = torch.roll(self.spe_w*self.excitations,shifts=-1,dims=2)
        inc_spe_w = torch.roll(self.spe_e*self.excitations,shifts=1,dims=2)
        inc_spe_s = torch.roll(self.spe_n*self.excitations,shifts=-1,dims=1)
        inc_spe_n = torch.roll(self.spe_s*self.excitations,shifts=1,dims=1)

        self.inc_spes = inc_spe_e*(1-self.spe_e)+inc_spe_w*(1-self.spe_w)+inc_spe_s*(1-self.spe_s)+inc_spe_n*(1-self.spe_n)
    
    def check_log_step(self):
        print('============================================================')
        print('============================================================')
        print('----------------------- STEP -----------------------')
        print('============================================================')
        print('============================================================')
        self.check_no_problem()

        self.compute_ord_excitations()
        self.compute_spe_excitations()
        print('----------------------- compute ord and spe excitations -----------------------')
        self.check_no_problem()

        self.compute_conf()
        print('----------------------- compute conf -----------------------')
        self.check_no_problem()
        self.compute_sens()
        print('----------------------- compute sens -----------------------')
        self.check_no_problem()

        self.excitations = ((self.inc_ords+self.inc_spes+self.inc_conf_ords+self.inc_conf_spes)>0).to(torch.uint8)
        self.excitations = torch.where((self.is_conf)*(1-self.conf_in)==1,0,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        # self.excitations = ((self.inc_ords+self.inc_spes)>0).to(torch.int)
        print('----------------------- compute excitations -----------------------')
        self.check_no_problem()

        self.compute_is_killed()
        print('----------------------- compute is killed -----------------------')
        self.check_no_problem()
        self.kill_dead()
        print('----------------------- kill dead -----------------------')
        self.check_no_problem()

        self.make_births()
        print('----------------------- make births -----------------------')
        self.check_no_problem()

        self.compute_is_ground()
        print('compute is ground -----------------------')
        self.check_no_problem()
    
    def step(self):
        self.compute_ord_excitations()
        self.compute_spe_excitations()

        self.compute_conf()
        self.compute_sens()

        self.excitations = ((self.inc_ords+self.inc_spes+self.inc_conf_ords+self.inc_conf_spes)>0).to(torch.uint8)
        self.excitations = torch.where((self.is_conf)*(1-self.conf_in)==1,0,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        self.compute_is_killed()
        self.kill_dead()
        self.make_births()

        self.compute_is_ground()
        self.check_no_problem()
    # def _check_pind(self):
    #     print('ord_e : ',self.ord_e[self.p_ind].item(),end=' ')
    #     print('ord_w :',self.ord_w[self.p_ind].item(),end=' ')
    #     print('ord_s :',self.ord_s[self.p_ind].item(),end=' ')
    #     print('ord_n :',self.ord_n[self.p_ind].item(),end=' ')

    #     print('spe_e :',self.spe_e[self.p_ind].item(),end=' ')
    #     print('spe_w :',self.spe_w[self.p_ind].item(),end=' ')
    #     print('spe_s :',self.spe_s[self.p_ind].item(),end=' ')
    #     print('spe_n :',self.spe_n[self.p_ind].item(),end=' ')

    #     print('conf_ :',self.is_conf[self.p_ind].item(),end=' ')
    #     print('sens_ :',self.is_sens[self.p_ind].item(),end=' ')
    #     print('sensv : ',self.sens_state[self.p_ind].item(),end=' ')

    #     print('is_ki : ',self.is_killed[self.p_ind].item(),end=' ')
    #     print('birth : ',self.births[self.p_ind].item(),end='')
    #     print('groun : ',self.is_ground[self.p_ind].item(),end='\n')

    def kill_dead(self):
        is_alive = (1-self.is_killed)
        # Try to batch this operation
        self.ord_e = self.ord_e*is_alive
        self.ord_w = self.ord_w*is_alive
        self.ord_s = self.ord_s*is_alive
        self.ord_n = self.ord_n*is_alive

        self.spe_e = self.spe_e*is_alive
        self.spe_w = self.spe_w*is_alive
        self.spe_s = self.spe_s*is_alive
        self.spe_n = self.spe_n*is_alive

        self.is_conf = self.is_conf*is_alive
        self.conf_in = self.conf_in*self.is_conf
        self.conf_out = self.conf_out*self.is_conf

        self.compute_is_ord()
        self.compute_is_spe()

        self.excitations = self.excitations*is_alive

    def check_no_problem(self):
        # Sum should not be bigger than one, otherwise we have overlapping states :
        problem_mask = self.ord_e+self.ord_w+self.ord_s+self.ord_n+self.spe_e+self.spe_w+self.spe_s+self.spe_n+self.is_conf+self.is_sens>1
        problem_ind = problem_mask.nonzero()
        if(problem_mask.any()):
            print('----------------------- FOUND PROBLEM -----------------------')
            print('ord_e : ',self.ord_e[problem_mask])
          
            print('ord_w :',self.ord_w[problem_mask])
            
            print('ord_s :',self.ord_s[problem_mask])
        
            print('ord_n :',self.ord_n[problem_mask])

            print('spe_e :',self.spe_e[problem_mask])

            print('spe_w :',self.spe_w[problem_mask])

            print('spe_s :',self.spe_s[problem_mask])

            print('spe_n :',self.spe_n[problem_mask])

            print('conf :',self.is_conf[problem_mask])

            print('sens :',self.is_sens[problem_mask])
            raise Exception('Found overlapping states at indices : ',problem_ind)
    
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

    @property
    def torch_worldmap(self):
        return self._worldmap.cpu()
    #################### MONTE CARLO INITIALIZATION FUNCTIONS ####################
    ############# MAKE SURE SET_STATE IS CALLED BEFORE RUNNING THESE ###############
    def compute_energy_density(self):
        """
            To be run AFTER set_state
        """

        aligned_weight = 1.
        opposite_weight = 1.

        energy_density = torch.zeros((1,*self.size), device=self.device)
        # This cannot be batched I think
        inc_ord_e = torch.roll(self.ord_w,shifts=-1,dims=2)
        inc_ord_w = torch.roll(self.ord_e,shifts=+1,dims=2)
        inc_ord_s = torch.roll(self.ord_n,shifts=-1,dims=1)
        inc_ord_n = torch.roll(self.ord_s,shifts=+1,dims=1)

        # equals one whenever at the current location we have opposing directions
        ord_opposites = (inc_ord_e*self.ord_e + inc_ord_w*self.ord_w + inc_ord_s*self.ord_s + inc_ord_n*self.ord_n) > 0
        ord_aligned = (inc_ord_e*self.ord_w + inc_ord_w*self.ord_e + inc_ord_s*self.ord_n + inc_ord_n*self.ord_s) > 0

        energy_density += ord_opposites*opposite_weight
        energy_density -= ord_aligned*aligned_weight

        # Now for special transmissions :
        inc_spe_e = torch.roll(self.spe_w,shifts=-1,dims=2)
        inc_spe_w = torch.roll(self.spe_e,shifts=+1,dims=2)
        inc_spe_s = torch.roll(self.spe_n,shifts=-1,dims=1)
        inc_spe_n = torch.roll(self.spe_s,shifts=+1,dims=1)

        # equals one whenever at the current location we have opposing directions
        spe_opposites = (inc_spe_e*(self.spe_e)+inc_spe_w*(self.spe_w)+inc_spe_s*(self.spe_s)+inc_spe_n*(self.spe_n))>0
        spe_aligned = (inc_spe_e*(self.spe_w)+inc_spe_w*(self.spe_e)+inc_spe_s*(self.spe_n)+inc_spe_n*(self.spe_s))>0

        energy_density += spe_opposites*opposite_weight
        energy_density -= spe_aligned*aligned_weight

        return energy_density
    
    def mcmc_step(self, beta, replace_prob=0.1):
        cur_state = self.get_state()
        cur_energy_density = self.compute_energy_density()

        rand_state = self.get_rando_para_state(self.p_ordinary, self.p_special, self.p_confluent)        
        proposed_state = torch.where(torch.rand(cur_state.shape,device=self.device)<replace_prob,rand_state,cur_state)

        self.set_state(proposed_state)
        new_energy_density = self.compute_energy_density()

        delta_energy = new_energy_density - cur_energy_density
        accept_prob = torch.exp(-beta*delta_energy)

        accept = torch.rand_like(accept_prob)<accept_prob
        self.set_state(torch.where(accept,proposed_state,cur_state))

    def run_mcmc(self, steps, beta, replace_prob=0.1):
        for _ in range(steps):
            self.mcmc_step(beta, replace_prob=replace_prob)



class BoolVonNeumann(Automaton):
    """
        Von Neumann cellular automaton

        Parameters :
        size : (H,W)
    """
    
    def __init__(self, size, device='cpu'):
        super().__init__(size, device=device)

        self.p_ordinary, self.p_special, self.p_confluent = .3,.3,.3
        state = self.get_rando_para_state(self.p_ordinary, self.p_special, self.p_confluent,.1)

        
        # Excitations :
        self.excitations = (torch.rand((1,*size), device=device)<0.5)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state,self.excitations)

        self._worldmap = torch.zeros((1,self.h,self.w), device=device, dtype=torch.int)

        self.to(device)
        self.p_ind = (0, 53, 70)
    def inj_excitations(self):
        self.excitations = (torch.rand((1,*self.size), device=self.device)<0.2).to(dtype=torch.uint8)

    def get_rando_para_state(self, p_ordinary=None, p_special=None, p_confluent=None, p_dead=0.,
                             state_size=None, batch_size=1):
        """
            Returns a random batch of states with probabilities given as parameters.

            Parameters :
            p_ordinary : Probability of ordinary transmission, if None, uses self
            p_special : Probability of special transmission, if None, uses self
            p_confluent : Probability of confluent, if None, uses self
            p_dead : Probability of dead cell
            state_size : size of the state. If None, uses self.size
            batch_size : Number of states to generate

            Return :
            state : (B,H,W) tensor of uint8
        """
        p_ordinary = self.p_ordinary if p_ordinary is None else p_ordinary
        p_special = self.p_special if p_special is None else p_special
        p_confluent = self.p_confluent if p_confluent is None else p_confluent
        state_size = self.size if state_size is None else state_size

        p_total = p_ordinary+p_special+p_confluent+p_dead

        p_ordinary /= p_total
        p_special /= p_total
        p_confluent /= p_total
        p_dead /= p_total

        state = torch.zeros((batch_size,*state_size),dtype=torch.int).to(self.device)

        # States random :
        uniform = torch.rand(state.shape).to(self.device)

        ordinary_locs = uniform < p_ordinary
        special_locs = (p_ordinary < uniform) & (uniform < p_ordinary + p_special)
        confluent_locs = (p_ordinary + p_special < uniform) & (uniform < p_ordinary + p_special + p_confluent)

        transmission_dirs = torch.randint_like(state, 0, 4)
        state = torch.where(ordinary_locs, transmission_dirs + 1, 0)
        state += torch.where(special_locs, transmission_dirs + 5, 0)
        state += torch.where(confluent_locs, 9, 0)

        return state

    def set_state(self, state, excitations = None):
        """
        	Fills all the tensor for each species given by the state, which should be a tensor of
            shape (1,H,W), with values between 0 and 9. If provided, sets excitations to the given
            (1,H,W) tensor of boolean/int(0,1) values. Otherwise, 0 excitations are set.
            TODO : provide option for random excitations.

            Args :
            state : (B,H,W) tensor of uint8
            excitations : (B,H,W) tensor of uint8 or booleans
        """
        if(excitations is None):
            excitations = torch.zeros_like(state, dtype=torch.bool).to(self.device)
        else :
            excitations = (excitations.to(torch.bool).to(self.device))
        assert state.shape == excitations.shape, "State and excitations should have the same shape"

        state = state.to(self.device)
        # Ordinary transmissions :
        self.ord_e = torch.where(state==1,True,False).to(torch.bool)
        self.ord_w = torch.where(state==2,True,False).to(torch.bool)
        self.ord_s = torch.where(state==3,True,False).to(torch.bool)
        self.ord_n = torch.where(state==4,True,False).to(torch.bool)
        
        # Special transmission :
        self.spe_e = torch.where(state==5,True,False).to(torch.bool)
        self.spe_w = torch.where(state==6,True,False).to(torch.bool)
        self.spe_s = torch.where(state==7,True,False).to(torch.bool)
        self.spe_n = torch.where(state==8,True,False).to(torch.bool)

        # Confluent :
        self.is_conf = torch.where(state==9,True,False).to(torch.bool)
        self.conf_in = torch.zeros_like(self.is_conf)
        self.conf_out = torch.zeros_like(self.is_conf)

        # Sensitized :
        self.is_sens = torch.where(state==10,True,False).to(torch.bool)
        self.sens_state = torch.where(self.is_sens,1,0).to(torch.uint8)
        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # Killed :
        self.is_killed = torch.zeros_like(state,dtype=torch.bool)

        # Init self.inc_ords and self.inc_spes
        self.compute_is_ord()
        self.compute_is_spe()
        # self.compute_ord_excitations()
        # self.compute_spe_excitations()
        self.compute_is_ground()
        
        self.excitations = excitations.to(self.device)



    def get_state(self):
        """
            Returns a tensor of shape (1,H,W), with values between 0 and 9.
        """
        state = torch.zeros((1,*self.size), device=self.device, dtype=torch.int)

        # Ordinary transmissions :
        state = torch.where(self.ord_e,1,state)
        state = torch.where(self.ord_w,2,state)
        state = torch.where(self.ord_s,3,state)
        state = torch.where(self.ord_n,4,state)
        
        # Special transmission :
        state = torch.where(self.spe_e,5,state)
        state = torch.where(self.spe_w,6,state)
        state = torch.where(self.spe_s,7,state)
        state = torch.where(self.spe_n,8,state)

        # Confluent :
        state = torch.where(self.is_conf,9,state)
        
        # TODO : ALLOW SAVING ALL THE DIFFERENT SENSITIZED STATES
        # Sensitized :
        state = torch.where(self.is_sens,10,state)

        return state

    def reset_state(self):
        state = torch.randint(0,10,(1,*self.size), device=self.device)

        mask = torch.zeros_like(state)
        mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]=0
        # mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]=torch.randint(6,8,(1,70,70), device=self.device)
    
        state[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35] = mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]
        
        # Excitations :
        excitations = (torch.rand((1,*self.size), device=self.device)<0.5).to(dtype=torch.uint8)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state,excitations)
    
    def make_state_bench(self):
        """
            Replaces the state with the benchmark state
        """
        state= torch.zeros((1,*self.size), device=self.device, dtype=torch.int)
        self.excitations = torch.zeros((1,*self.size), device=self.device, dtype=torch.int)

        bench_state, bench_excit = get_sens_benchmark(True)
        state[0,2:9+2,2:5+2] = bench_state.to(self.device)
        self.excitations[0,2:9+2,2:5+2] = bench_excit.to(self.device)

        return state
    
    def compute_is_ord(self):
        self.is_ord = (self.ord_e|self.ord_w|self.ord_s|self.ord_n)
    
    def compute_is_spe(self):
        self.is_spe = (self.spe_e|self.spe_w|self.spe_s|self.spe_n)
    

    def compute_is_killed(self):
        self.compute_is_ord()
        self.compute_is_spe()

        self.is_killed = (self.is_ord&self.inc_spes | self.is_spe&self.inc_ords | self.is_conf&self.inc_spes)

    def compute_is_ground(self):
        self.is_ground = (~self.is_ord)&(~self.is_spe)&(~self.is_conf)&(~self.is_sens)


    def compute_conf(self):
        self.compute_inhibitions()

        conf_act = self.conf_out&self.is_conf
        inc_conf_e = torch.roll(conf_act,shifts=-1,dims=2)
        inc_conf_w = torch.roll(conf_act,shifts=+1,dims=2)
        inc_conf_s = torch.roll(conf_act,shifts=-1,dims=1)
        inc_conf_n = torch.roll(conf_act,shifts=+1,dims=1)
        
        # This looses an excitation on top of a confluent if put manually, but it's acceptable
        self.inc_conf_ords = self.is_ord&(inc_conf_e&(~self.ord_e)|inc_conf_w&(~self.ord_w)|inc_conf_s&(~self.ord_s)|inc_conf_n&(~self.ord_n))
        self.inc_conf_spes = self.is_spe&(inc_conf_e&(~self.spe_e)|inc_conf_w&(~self.spe_w)|inc_conf_s&(~self.spe_s)|inc_conf_n&(~self.spe_n))

        self.conf_out = self.conf_in&self.is_conf
        self.conf_in = (~self.inh)&self.inc_ords&self.is_conf

    
    def compute_sens(self):
        inc_exc = (self.inc_ords|self.inc_spes)

        self.is_sens = torch.where((self.is_ground)&inc_exc,True,self.is_sens)

        self.sens_state = ((self.sens_state << 1)+inc_exc)*self.is_sens # Booleans are casted to ints there !

        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # I think this is very inefficient also, not sure how to batch it
        self.births = torch.where((self.sens_state==0b10000),1,self.births) # East
        self.births = torch.where((self.sens_state==0b1001),2,self.births) # West
        self.births = torch.where((self.sens_state==0b1010),3,self.births) # South
        self.births = torch.where((self.sens_state==0b10001),4,self.births) # North

        self.births = torch.where((self.sens_state==0b1011),1+4,self.births) # East Special
        self.births = torch.where((self.sens_state==0b1101),2+4,self.births) # West Special
        self.births = torch.where((self.sens_state==0b1110),3+4,self.births) # South Special
        self.births = torch.where((self.sens_state==0b1100),4+4,self.births) # North Special

        self.births = torch.where((self.sens_state==0b1111),9,self.births) # Confluent

        self.is_sens = torch.where(self.births>0,False,self.is_sens)
        self.sens_state = self.sens_state*self.is_sens # Casts to int
    
    def make_births(self):
        # Extinguish remaining excitations on birthing cells
        self.excitations = torch.where(self.births>0,False,self.excitations) 

        # This cannot be batched I think
        self.ord_e = torch.where(self.births==1,True,self.ord_e)
        self.ord_n = torch.where(self.births==4,True,self.ord_n)
        self.ord_w = torch.where(self.births==2,True,self.ord_w)
        self.ord_s = torch.where(self.births==3,True,self.ord_s)

        self.spe_e = torch.where(self.births==5,True,self.spe_e)
        self.spe_n = torch.where(self.births==8,True,self.spe_n)
        self.spe_w = torch.where(self.births==6,True,self.spe_w)
        self.spe_s = torch.where(self.births==7,True,self.spe_s)

        self.is_conf = torch.where(self.births==9,True,self.is_conf)

        self.births = torch.zeros_like(self.births)
        # Recompute which the 'is' thingies
        self.compute_is_ord()
        self.compute_is_spe()

    def compute_ord_excitations(self):
        # This cannot be batched I think
        inc_ord_e = torch.roll(self.ord_w&self.excitations,shifts=-1,dims=2)
        inc_ord_w = torch.roll(self.ord_e&self.excitations,shifts=1,dims=2)
        inc_ord_s = torch.roll(self.ord_n&self.excitations,shifts=-1,dims=1)
        inc_ord_n = torch.roll(self.ord_s&self.excitations,shifts=1,dims=1)

        self.inc_ords = inc_ord_e&(~self.ord_e)|inc_ord_w&(~self.ord_w)|inc_ord_s&(~self.ord_s)|inc_ord_n&(~self.ord_n)

    def compute_inhibitions(self):
        # This cannot be batched I think
        inc_inh_e = torch.roll(self.ord_w&(~self.excitations),shifts=-1,dims=2)
        inc_inh_w = torch.roll(self.ord_e&(~self.excitations),shifts=1,dims=2)
        inc_inh_s = torch.roll(self.ord_n&(~self.excitations),shifts=-1,dims=1)
        inc_inh_n = torch.roll(self.ord_s&(~self.excitations),shifts=1,dims=1)

        self.inh = (inc_inh_e|inc_inh_w|inc_inh_s|inc_inh_n)

    def compute_spe_excitations(self):
        """
            Computes the excitations of the special particles, and returns the purged incoming excitations.
        """
        # This cannot be batched I think
        inc_spe_e = torch.roll(self.spe_w&self.excitations,shifts=-1,dims=2)
        inc_spe_w = torch.roll(self.spe_e&self.excitations,shifts=1,dims=2)
        inc_spe_s = torch.roll(self.spe_n&self.excitations,shifts=-1,dims=1)
        inc_spe_n = torch.roll(self.spe_s&self.excitations,shifts=1,dims=1)

        self.inc_spes = inc_spe_e&(~self.spe_e)|inc_spe_w&(~self.spe_w)|inc_spe_s&(~self.spe_s)|inc_spe_n&(~self.spe_n)
    
    def check_log_step(self):
        print('============================================================')
        print('============================================================')
        print('----------------------- STEP -----------------------')
        print('============================================================')
        print('============================================================')
        self.check_no_problem()

        self.compute_ord_excitations()
        self.compute_spe_excitations()
        print('----------------------- compute ord and spe excitations -----------------------')
        self.check_no_problem()

        self.compute_conf()
        print('----------------------- compute conf -----------------------')
        self.check_no_problem()
        self.compute_sens()
        print('----------------------- compute sens -----------------------')
        self.check_no_problem()

        self.excitations = ((self.inc_ords|self.inc_spes+self.inc_conf_ords+self.inc_conf_spes)>0).to(torch.uint8)
        self.excitations = torch.where((self.is_conf)*(1-self.conf_in)==1,0,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        # self.excitations = ((self.inc_ords+self.inc_spes)>0).to(torch.int)
        print('----------------------- compute excitations -----------------------')
        self.check_no_problem()

        self.compute_is_killed()
        print('----------------------- compute is killed -----------------------')
        self.check_no_problem()
        self.kill_dead()
        print('----------------------- kill dead -----------------------')
        self.check_no_problem()

        self.make_births()
        print('----------------------- make births -----------------------')
        self.check_no_problem()

        self.compute_is_ground()
        print('compute is ground -----------------------')
        self.check_no_problem()
    
    def step(self):
        self.compute_ord_excitations()
        self.compute_spe_excitations()

        self.compute_conf()
        self.compute_sens()

        self.excitations = (self.inc_ords|self.inc_spes|self.inc_conf_ords|self.inc_conf_spes)
        # self.excitations = torch.where((self.is_conf)&(~self.conf_in),False,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        self.compute_is_killed()
        self.kill_dead()
        self.make_births()

        self.compute_is_ground()
        self.check_no_problem()
    # def _check_pind(self):
    #     print('ord_e : ',self.ord_e[self.p_ind].item(),end=' ')
    #     print('ord_w :',self.ord_w[self.p_ind].item(),end=' ')
    #     print('ord_s :',self.ord_s[self.p_ind].item(),end=' ')
    #     print('ord_n :',self.ord_n[self.p_ind].item(),end=' ')

    #     print('spe_e :',self.spe_e[self.p_ind].item(),end=' ')
    #     print('spe_w :',self.spe_w[self.p_ind].item(),end=' ')
    #     print('spe_s :',self.spe_s[self.p_ind].item(),end=' ')
    #     print('spe_n :',self.spe_n[self.p_ind].item(),end=' ')

    #     print('conf_ :',self.is_conf[self.p_ind].item(),end=' ')
    #     print('sens_ :',self.is_sens[self.p_ind].item(),end=' ')
    #     print('sensv : ',self.sens_state[self.p_ind].item(),end=' ')

    #     print('is_ki : ',self.is_killed[self.p_ind].item(),end=' ')
    #     print('birth : ',self.births[self.p_ind].item(),end='')
    #     print('groun : ',self.is_ground[self.p_ind].item(),end='\n')

    def kill_dead(self):
        is_alive = ~self.is_killed
        # Try to batch this operation
        self.ord_e = self.ord_e&is_alive
        self.ord_w = self.ord_w&is_alive
        self.ord_s = self.ord_s&is_alive
        self.ord_n = self.ord_n&is_alive

        self.spe_e = self.spe_e&is_alive
        self.spe_w = self.spe_w&is_alive
        self.spe_s = self.spe_s&is_alive
        self.spe_n = self.spe_n&is_alive

        self.is_conf = self.is_conf&is_alive
        self.conf_in = self.conf_in&self.is_conf
        self.conf_out = self.conf_out&self.is_conf

        self.compute_is_ord()
        self.compute_is_spe()

        self.excitations = self.excitations&is_alive

    def check_no_problem(self):
        # Sum should not be bigger than one, otherwise we have overlapping states :
        problem_mask = (self.ord_e.int()+self.ord_w.int()+self.ord_s.int()+self.ord_n.int()\
                        +self.spe_e.int()+self.spe_w.int()+self.spe_s.int()+self.spe_n.int()\
                        +self.is_conf.int()+self.is_sens.int())>1
        
        problem_ind = problem_mask.nonzero()
        if(problem_mask.any()):
            print('----------------------- FOUND PROBLEM -----------------------')
            print('ord_e : ',self.ord_e[problem_mask])
          
            print('ord_w :',self.ord_w[problem_mask])
            
            print('ord_s :',self.ord_s[problem_mask])
        
            print('ord_n :',self.ord_n[problem_mask])

            print('spe_e :',self.spe_e[problem_mask])

            print('spe_w :',self.spe_w[problem_mask])

            print('spe_s :',self.spe_s[problem_mask])

            print('spe_n :',self.spe_n[problem_mask])

            print('conf :',self.is_conf[problem_mask])

            print('sens :',self.is_sens[problem_mask])
            raise Exception('Found overlapping states at indices : ',problem_ind)
    
    def draw(self):
        self._worldmap = torch.zeros_like(self._worldmap)
        self._worldmap = torch.where(self.ord_e,1,self._worldmap)
        self._worldmap = torch.where(self.ord_w,2,self._worldmap)
        self._worldmap = torch.where(self.ord_s,3,self._worldmap)
        self._worldmap = torch.where(self.ord_n,4,self._worldmap)
        self._worldmap = torch.where(self.spe_e,5,self._worldmap)
        self._worldmap = torch.where(self.spe_w,6,self._worldmap)
        self._worldmap = torch.where(self.spe_s,7,self._worldmap)
        self._worldmap = torch.where(self.spe_n,8,self._worldmap)
        self._worldmap = torch.where(self.is_conf,9,self._worldmap)
        self._worldmap = torch.where(self.is_sens,10,self._worldmap)
    
    @property
    def worldmap(self):
        # OVERRIDE WORLDMAP TO CONTAIN TYPES OF PARTICLES
        return self._worldmap.permute(2,1,0).cpu().numpy(), self.excitations.int().permute(2,1,0).cpu().numpy(), self.conf_out.int().permute(2,1,0).cpu().numpy()

    @property
    def torch_worldmap(self):
        return self._worldmap.cpu()