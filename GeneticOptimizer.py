from Automaton import *
from torchenhanced.util import showTens
import statistics, random, torch
from tqdm import tqdm
import numpy as np



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

    def __init__(self,size,square_size,population_size,simulation_steps,
                 survival_rate,incoming_rate,device='cpu'):
        self.automaton = VonNeumann(size,device=device)

        self.population_size = population_size
        self.simulation_steps = simulation_steps

        self.size = size
        self.square_size = square_size

        self.initial_excitation = (torch.rand(self.automaton.excitations.shape)<0.5).to(dtype=torch.uint8)

        self.states=[]
        for _ in tqdm(range(population_size)):
            self.states.append(self.get_random_state())
        self.states = [self.get_random_state() for _ in range(population_size)]


    def get_random_state(self):
        self.automaton.set_state(self.automaton.get_rando_para_state(0.3,0.3,0.3))
        self.automaton.run_mcmc(10,2,replace_prob=0.1)
        H,W = self.size
        h,w = self.square_size

        x = torch.linspace(0, W-1, W)
        y = torch.linspace(0, H-1, H)
        xx, yy = torch.meshgrid(x, y)

        # Calculate the top-left and bottom-right coordinates of the centered square
        start_h = (H - h) // 2
        start_w = (W - w) // 2
        end_h = start_h + h
        end_w = start_w + w

        # Create the mask based on the condition
        mask = ((xx >= start_w) & (xx < end_w) & (yy >= start_h) & (yy < end_h)).float()

        return self.automaton.get_state()*mask[None,:,:]
    
    def fitness_num_states_0(self,state):
        self.automaton.set_state(state)
        self.automaton.excitations = self.initial_excitation.clone()

        for _ in range(self.simulation_steps):
            self.automaton.step()
        
        self.automaton.compute_is_ground()

        # showTens(self.automaton.is_ground)

        return (1-self.automaton.is_ground).to(dtype=torch.float32).mean().item()

    def generate_child(self,parent1,parent2):
        """
            Generates a child from two parents.
            The child is a random mix of the two parents.
        """
        child_state = torch.where(torch.linspace(0,1,self.size[0])[None,None,:]<0.5,parent1,parent2)

        return child_state

    def draw_state(self, state):
        self.automaton.set_state(state)
        self.automaton.draw()
        # print(f'Shape : {torch.tensor(self.automaton.worldmap).shape}')
        showTens(self.automaton.torch_worldmap)
    
    def evolve(self,num_generations):
        for k in range(num_generations):
            fitnesses = [self.fitness_num_states_0(state) for state in self.states]
            thresh_fitness = np.percentile(fitnesses,75)
            mean_fitness = statistics.mean(fitnesses)
            max_index = fitnesses.index(max(fitnesses))

            if k+1==num_generations:
                self.draw_state(self.states[max_index])
            
            print(f'Gen {k} : {thresh_fitness:}, {mean_fitness:}''Mean fitness: ',mean_fitness,' Median fitness: ',thresh_fitness)

            surviving_states = [state for (state, fitness) in zip(self.states, fitnesses) if fitness >= thresh_fitness]
            surviving_states.extend([self.get_random_state() for _ in range(self.population_size//10)])
            next_states = [self.generate_child(random.choice(surviving_states),random.choice(surviving_states)) for _ in range(self.population_size)]
            self.states = next_states

if __name__=='__main__':
    geno = GeneticOptimizer((100,100),(30,30),40,30,0.5,0.5,device='cpu')

    geno.evolve(300)