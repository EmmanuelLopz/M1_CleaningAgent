# Code to represent a cleaning automata in a MxN space
# Code by Eduardo Mora - A01799440 and Emmanuel Lopez - A01666331
# Last Modification: 11/11/2025

import mesa
import numpy as np
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class CleaningAgent(mesa.Agent):
    def __init__(self, model, coordination_distance=5):
        super().__init__(model)
        self.movements = 0
        self.coordination_distance = coordination_distance
        
        self.action_space = ['move_to_dirty', 'stay', 'move_to_clean']
        self.current_action = None
        
        self.utility_move_dirty = 10.0
        self.utility_stay = 0.0
        self.utility_move_clean = -2.0
        
        self.payoff_different_cells = 5.0
        self.payoff_same_cell_collision = -15.0
        self.payoff_default = 0.0
        
        self.neighbors = []
        self.messages_received = {}
        self.messages_to_send = {}

        self.target_pos = None
        
    def find_neighbors(self):
        """Find neighbors within coordination distance (Manhattan distance)"""
        self.neighbors = []
        for agent in self.model.agents:
            if agent.unique_id != self.unique_id:
                manhattan_dist = abs(agent.pos[0] - self.pos[0]) + abs(agent.pos[1] - self.pos[1])
                if manhattan_dist <= self.coordination_distance:
                    self.neighbors.append(agent)
    
    def compute_local_utility(self, action):
        if action == 'move_to_dirty':
            return self.utility_move_dirty
        elif action == 'stay':
            return self.utility_stay
        elif action == 'move_to_clean':
            return self.utility_move_clean
        return 0.0
    
    def compute_neighbor_payoff(self, my_action, neighbor_action, my_target, neighbor_target):
        if my_target != neighbor_target:
            return self.payoff_different_cells
        
        if my_target == neighbor_target and (my_action != 'stay' or neighbor_action != 'stay'):
            return self.payoff_same_cell_collision
        
        return self.payoff_default
    
    def get_candidate_positions(self, action):
        x, y = self.pos
        m, n = self.model.grid.width, self.model.grid.height
        
        if action == 'stay':
            return [(x, y)]
        
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < m and 0 <= new_y < n:
                    if action == 'move_to_dirty' and self.model.dirtyGrid[new_x][new_y] == 1:
                        candidates.append((new_x, new_y))
                    elif action == 'move_to_clean' and self.model.dirtyGrid[new_x][new_y] == 0:
                        candidates.append((new_x, new_y))
        
        if not candidates and action != 'stay':
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < m and 0 <= new_y < n:
                        candidates.append((new_x, new_y))
        
        return candidates if candidates else [(x, y)]
    
    def max_plus_iteration(self):
        self.messages_to_send = {}
        
        for neighbor in self.neighbors:
            message = {}
            
            for neighbor_action in self.action_space:
                max_value = float('-inf')
                
                for my_action in self.action_space:
                    my_candidates = self.get_candidate_positions(my_action)
                    neighbor_candidates = neighbor.get_candidate_positions(neighbor_action)
                    
                    if my_candidates and neighbor_candidates:
                        my_target = self.random.choice(my_candidates)
                        neighbor_target = self.random.choice(neighbor_candidates)
                        
                        utility = self.compute_local_utility(my_action)
                        payoff = self.compute_neighbor_payoff(my_action, neighbor_action, my_target, neighbor_target)
                        
                        message_sum = 0.0
                        for other_neighbor_id, other_messages in self.messages_received.items():
                            if other_neighbor_id != neighbor.unique_id and my_action in other_messages:
                                message_sum += other_messages[my_action]
                        
                        total_value = utility + payoff + message_sum
                        
                        if total_value > max_value:
                            max_value = total_value
                
                message[neighbor_action] = max_value
            
            self.messages_to_send[neighbor.unique_id] = message
    
    def send_messages(self):
        for neighbor in self.neighbors:
            if neighbor.unique_id in self.messages_to_send:
                neighbor.messages_received[self.unique_id] = self.messages_to_send[neighbor.unique_id]
    
    def select_best_action(self):
        best_action = None
        best_value = float('-inf')
        best_target = None
        
        for action in self.action_space:
            utility = self.compute_local_utility(action)
            message_sum = 0.0
            for neighbor_messages in self.messages_received.values():
                if action in neighbor_messages:
                    message_sum += neighbor_messages[action]
            
            total_value = utility + message_sum
            
            if total_value > best_value:
                best_value = total_value
                best_action = action
                candidates = self.get_candidate_positions(action)
                best_target = self.random.choice(candidates) if candidates else self.pos
        
        self.current_action = best_action
        self.target_pos = best_target
        
        return best_action, best_target
    
    def execute_action(self):
        x, y = self.pos

        if self.model.dirtyGrid[x][y] == 1:
            self.model.dirtyGrid[x][y] = 0
            return

        if self.current_action == 'stay':
            pass
        elif self.target_pos and self.target_pos != self.pos:
            self.model.grid.move_agent(self, self.target_pos)
            self.movements += 1
    
    def step(self):
        """Agent's behavior per step with Max-Plus coordination"""
        # This method will be called as part of the coordination cycle
        # The actual coordination happens in the model's step method
        pass

class RoomToClean(mesa.Model):
    def __init__(self, m, n, numAgent, num_dirty_cells, maxTime, coordination_distance=5, max_plus_iterations=10):
        super().__init__()
        self.numAgent = numAgent
        self.maxRounds = maxTime
        self.actualWeight = 0
        self.coordination_distance = coordination_distance
        self.max_plus_iterations = max_plus_iterations

        self.grid = MultiGrid(m, n, torus=False)
        
        self.dirtyGrid = np.zeros((m, n), dtype=int)
        
        total_cells = m * n
        dirty_indices = np.random.choice(total_cells, size=num_dirty_cells, replace=False)
        for idx in dirty_indices:
            x = idx // n
            y = idx % n
            self.dirtyGrid[x][y] = 1
        
        clean_positions = [(x, y) for x in range(m) for y in range(n) if self.dirtyGrid[x][y] == 0]
        
        agent_positions = self.random.sample(clean_positions, min(numAgent, len(clean_positions)))
        for pos in agent_positions:
            agent = CleaningAgent(self, coordination_distance=coordination_distance)
            self.grid.place_agent(agent, pos=pos)

        self.datacollector = DataCollector(
            model_reporters={
                "CleanedCells": lambda m: np.sum(m.dirtyGrid == 0),
                "DirtyCells": lambda m: np.sum(m.dirtyGrid == 1),
                "PercentageCleaned": lambda m: (1.0 - np.mean(m.dirtyGrid)) * 100
            },
            agent_reporters={
                "Movements": lambda a: a.movements,
                "Action": lambda a: a.current_action
            }
        )

    def step(self):
        self.datacollector.collect(self)
        
        if self.actualWeight >= self.maxRounds or np.sum(self.dirtyGrid) == 0:
            self.running = False
            return
        
        for agent in self.agents:
            agent.find_neighbors()

        for iteration in range(self.max_plus_iterations):
            for agent in self.agents:
                agent.max_plus_iteration()
    
            for agent in self.agents:
                agent.send_messages()

        for agent in self.agents:
            agent.select_best_action()

        agent_list = list(self.agents)
        self.random.shuffle(agent_list)
        for agent in agent_list:
            agent.execute_action()

        for agent in self.agents:
            agent.messages_received = {}
            agent.messages_to_send = {}
        
        self.actualWeight += 1

if __name__ == "__main__":
    m, n = 20, 30
    num_dirty_cells = 100
    maxTime = 500
    coordination_distance = 10
    max_plus_iterations = 10

    robot_counts = [1, 2, 4, 8, 16]
    
    print("=" * 80)
    print("CLEANING ROBOTS WITH COORDINATION GRAPH + MAX-PLUS MESSAGE PASSING")
    print("=" * 80)
    print(f"Grid Size: {m}x{n} ({m*n} total cells)")
    print(f"Dirty Cells: {num_dirty_cells}")
    print(f"Max Time Steps: {maxTime}")
    print(f"Coordination Distance: {coordination_distance} (Manhattan)")
    print(f"Max-Plus Iterations: {max_plus_iterations}")
    print("=" * 80)
    
    results = []
    
    for numAgent in robot_counts:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {numAgent} Robot{'s' if numAgent > 1 else ''}")
        print(f"{'='*80}")
        
        model = RoomToClean(
            m=m, 
            n=n, 
            numAgent=numAgent, 
            num_dirty_cells=num_dirty_cells, 
            maxTime=maxTime,
            coordination_distance=coordination_distance,
            max_plus_iterations=max_plus_iterations
        )
        
        print(f"Starting simulation...")
        model.run_model()
        
        # Collect results
        requiredTime = model.actualWeight
        finalDirtyCells = np.sum(model.dirtyGrid)
        finalCleanedCells = np.sum(model.dirtyGrid == 0)
        finalPercentageCleaned = (1.0 - np.mean(model.dirtyGrid)) * 100
        
        dataAgent = model.datacollector.get_agent_vars_dataframe()
        if requiredTime > 0:
            last_step_index = requiredTime - 1
            finalMovements = dataAgent.xs(last_step_index, level="Step")["Movements"].sum()
        else:
            finalMovements = 0
        
        success = finalDirtyCells == 0 and requiredTime <= maxTime
        
        print(f"\n{'─'*80}")
        print(f"RESULTS:")
        print(f"{'─'*80}")
        print(f"Time Steps Used: {requiredTime}/{maxTime}")
        print(f"Success: {'✓ YES' if success else '✗ NO'}")
        print(f"Cleaned Cells: {finalCleanedCells}/{m*n} ({finalPercentageCleaned:.2f}%)")
        print(f"Remaining Dirty Cells: {finalDirtyCells}")
        print(f"Total Movements: {finalMovements}")
        print(f"Average Movements per Robot: {finalMovements/numAgent:.2f}")
        print(f"Efficiency (Cells/Movement): {(finalCleanedCells - (m*n - num_dirty_cells))/max(finalMovements, 1):.2f}")
        print(f"{'─'*80}")
        
        results.append({
            'robots': numAgent,
            'time_steps': requiredTime,
            'success': success,
            'cleaned_cells': finalCleanedCells,
            'dirty_cells': finalDirtyCells,
            'total_movements': finalMovements,
            'avg_movements': finalMovements/numAgent,
            'efficiency': (finalCleanedCells - (m*n - num_dirty_cells))/max(finalMovements, 1)
        })
    
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'='*80}")
    print(f"{'Robots':<10}{'Time':<10}{'Success':<12}{'Cleaned':<12}{'Movements':<15}{'Efficiency':<12}")
    print(f"{'─'*80}")
    for r in results:
        success_str = "✓" if r['success'] else "✗"
        print(f"{r['robots']:<10}{r['time_steps']:<10}{success_str:<12}{r['cleaned_cells']}/{m*n:<7}{r['total_movements']:<15}{r['efficiency']:.2f}")
    print(f"{'='*80}")

