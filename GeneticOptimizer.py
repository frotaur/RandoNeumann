from Automaton import *
from torchenhanced.util import showTens
import statistics, random, torch
from tqdm import tqdm
import numpy as np
import time


class GeneticOptimizer:
    """
        Class to optimize an Automaton using a genetic algorithm.
    
        Args:
        size : size of Automaton
        square_size : size of modifiable square in Automaton
        population_size : size of population
        simulation_steps : number of steps to simulate
        survival_rate : percentage of population to survive
        incoming_rate : percentage of new population to be added to children
    """

    def __init__(self,size,square_size,population_size,simulation_steps,mutation_rate=0.03,device='cpu'):
        self.automaton = VonNeumann(size,device=device)

        self.population_size = population_size
        self.simulation_steps = simulation_steps

        self.size = size
        self.square_size = square_size
        self.square_mask = self.get_square_mask()

        self.initial_excitation = (torch.rand(self.automaton.excitations.shape)<0.5).to(dtype=torch.uint8).to(self.automaton.device)
        torch.save(self.initial_excitation,'initial_excitation.pt')
        self.states=[]
        for _ in tqdm(range(population_size)):
            self.states.append(self.get_random_state())
        self.states = [self.get_random_state() for _ in range(population_size)]

        self.mutation_rate = mutation_rate

        self.device = device
    

    def get_square_mask(self):
        H,W = self.size
        h,w = self.square_size

        x = torch.linspace(0, W-1, W,device=self.automaton.device)
        y = torch.linspace(0, H-1, H,device=self.automaton.device)
        xx, yy = torch.meshgrid(x, y)

        # Calculate the top-left and bottom-right coordinates of the centered square
        start_h = (H - h) // 2
        start_w = (W - w) // 2
        end_h = start_h + h
        end_w = start_w + w

        # Create the mask based on the condition
        mask = ((xx >= start_w) & (xx < end_w) & (yy >= start_h) & (yy < end_h)).float()

        return mask
    
    def get_random_state(self):
        self.automaton.set_state(self.automaton.get_rando_para_state(0.3,0.3,0.3))
        # self.automaton.run_mcmc(1,2,replace_prob=0.1)

        return self.automaton.get_state()*self.square_mask[None,:,:]
    
    def fitness_mover(self,state):
        eval_period = 20

        self.automaton.set_state(state)
        for _ in range(self.simulation_steps):
            self.automaton.step()
        
        excitation_table = []
        for _ in range(eval_period):
            self.automaton.step()
            excitation_table.append(self.automaton.excitations.clone())

        excitation_tensor = torch.stack(excitation_table,dim=0) # (eval_period,H,W)
        score_tensor = torch.zeros_like(excitation_tensor) # (eval_period,H,W)
        for i in range(1, eval_period - 1):
            max_before = excitation_tensor[:i, :, :].max(dim=0)[0]
            max_after = excitation_tensor[i:, :, :].max(dim=0)[0]
            score_tensor += (max_after - max_before).clamp(min=0)

        return score_tensor.mean().item()
    
    def fitness_num_states_0(self,state):

        self.automaton.set_state(state)
        self.automaton.excitations = self.initial_excitation.clone()


        for _ in range(self.simulation_steps):
            self.automaton.step()
        
        # self.automaton.compute_is_ground()

        return (self.automaton.is_conf).to(dtype=torch.float32).mean().item()

    def repro_mask(self, num_centers=8, r=None):
        """
            Compute a 0/1 mask with big chunks, that determines from which parent the child state 
            will inherit the pixel.

            returns a (1, h, w) tensor
        """
        h,w = self.size

        if r is None: r = (w + h) / 10 # a bit hacky
        xx, yy = torch.meshgrid(torch.linspace(0, w - 1, w), torch.linspace(0, h - 1, h), indexing='xy')
        field = torch.zeros(w, h)
        for i in range(num_centers):
            cx, cy = random.random() * w, random.random() * h
            sqxx, sqyy = (xx - cx) ** 2, (yy - cy) ** 2
            sqd = sqxx + sqyy
            field += (2 * random.random() - 1) * torch.exp(-sqd / (r ** 2))

        return field>=0. # [h, w]

    def generate_child(self,parent1,parent2):
        """
            Generates a child from two parents.
            The child is a random mix of the two parents.
        """

        # reproduction by crossover
        child_state = torch.where(torch.full(parent1.shape,True),parent1,parent2)
        
        # Mutations
        child_state = torch.where(torch.rand(self.size,device=self.device)<self.mutation_rate,self.get_random_state(),child_state)
       
        return child_state


    def draw_state(self, state):
        self.automaton.set_state(state)
        self.automaton.draw()
        # print(f'Shape : {torch.tensor(self.automaton.worldmap).shape}')
        showTens(self.automaton.torch_worldmap)
    
    def evolve(self,num_generations):
        num_survive = int(self.population_size*0.1)
        num_reprod = int(self.population_size*0.2)
        num_children = self.population_size-num_survive
        for k in range(num_generations):
            t1 = time.time()
            fitnesses = []
            for state in tqdm(self.states):
                fitnesses.append(self.fitness_num_states_0(state))
        
            sorted_indices = np.argsort(fitnesses)[::-1]

            mean_fitness = statistics.mean(fitnesses)
            max_index = fitnesses.index(max(fitnesses))

            if k+1==num_generations:
                self.draw_state(self.states[max_index])
            torch.save(self.states[max_index],'best_state.pt')

            print(f'Gen {k} : {mean_fitness:}''Mean fitness: ',mean_fitness)

            surviving_states = [self.states[k] for k in sorted_indices[:num_survive]]
            reproducing_states = [self.states[k] for k in sorted_indices[:num_reprod]]

            next_child_states =[self.generate_child(random.choice(reproducing_states),random.choice(reproducing_states)) for _ in range(num_children)]
            
            self.states = surviving_states+next_child_states
            assert len(self.states)==self.population_size

if __name__=='__main__':
    geno = GeneticOptimizer((100,100),(30,30),40,150,0.03,device='cpu')

    geno.evolve(100)