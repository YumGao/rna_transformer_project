# +
import math
import random

from tools import energy


# -

class simulated_annealing:
    def __init__(self, initial_structure, objective_function, initial_temperature=100, cooling_rate=0.95, max_iterations=1000):

        self.initial_structure = initial_structure
        self.objective_function = objective_function
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def perturb_structure(self, current_structure):

        new_structure = current_structure.copy()
        perturbation_type = random.choice(['single_replace', 'multi_replace', 'local_swap'])

        if perturbation_type == 'single_replace':
            index = random.randint(0, len(new_structure) - 1)
            bases = ["A", "C", "G", "U"]
            current_base = new_structure[index]
            available_bases = [base for base in bases if base != current_base]
            new_structure[index] = random.choice(available_bases)
        elif perturbation_type == 'multi_replace':
            num_replace = random.randint(1, min(5, len(new_structure)))
            indices = random.sample(range(len(new_structure)), num_replace)
            bases = ["A", "C", "G", "U"]
            for index in indices:
                current_base = new_structure[index]
                available_bases = [base for base in bases if base != current_base]
                new_structure[index] = random.choice(available_bases)
        elif perturbation_type == 'local_swap':
            start = random.randint(0, len(new_structure) - 2)
            end = random.randint(start + 1, len(new_structure) - 1)
            new_structure[start:end] = new_structure[start:end][::-1]

        return new_structure

    def simulate(self):
        """
        run simulate_annealing
        """
        current_structure = self.initial_structure
        current_energy = self.objective_function(current_structure)
        best_structure = current_structure
        best_energy = current_energy
        temperature = self.initial_temperature

        for _ in range(self.max_iterations):
            new_structure = self.perturb_structure(current_structure)
            new_energy = self.objective_function(new_structure)

            delta_energy = new_energy - current_energy

            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_structure = new_structure
                current_energy = new_energy

            if current_energy < best_energy:
                best_structure = current_structure
                best_energy = current_energy

            temperature *= self.cooling_rate

        return best_structure, best_energy


initial_structure = ['A', 'U', 'A', 'C', 'G', 'C', 'A', 'A', 'G', 'G', 'A', 'C', 'C', 'G', 'A', 'U', 'C', 'G', 'G', 'U']
simulator = simulated_annealing(initial_structure, energy.minimum_free_energy)
best_structure, best_energy = simulator.simulate()



