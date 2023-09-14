import time
import os
import Automaton


size = (10000,10000)
auto = Automaton.VonNeumann(size,device='cuda')

def run_automaton(auto, frame_avg=100):
    frame_times = []

    for i in range(1, 100):
        start_time = time.time()
        for _ in range(0, frame_avg):
            auto.step()

        # Calculate time taken for this frame bunch
        end_time = time.time()
        frame_time = end_time - start_time
        
        fps = (1.0 / frame_time)*frame_avg
        
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Average FPS : {fps:.2f}")

run_automaton(auto,10)