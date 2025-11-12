# Code to represent a cleaning automata in a MxN space
# Code by Eduardo Mora - A01799440
# Last Modification: 11/11/2025

import mesa
import numpy as np
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class CleaningAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.movements = 0

    def step(self):
        x, y = self.pos
        cellContent = self.model.dirtyGrid[x][y]

        if cellContent == 1:
            self.model.dirtyGrid[x][y] = 0
        else:
            neighbors = self.model.grid.get_neighborhood(
                pos=self.pos,
                moore=True,
                include_center=False
            )
            if neighbors:
                newPos = self.random.choice(neighbors)
                self.model.grid.move_agent(self, newPos)
                self.movements += 1

class RoomToClean(mesa.Model):
    def __init__(self, m, n, numAgent, dirtyPercentage, maxTime):
        super().__init__()
        self.numAgent = numAgent
        self.maxRounds = maxTime
        self.actualWeight = 0

        self.grid = MultiGrid(m, n, torus=False)
        self.dirtyGrid = np.where(np.random.rand(m, n) < dirtyPercentage, 1, 0)

        for _ in range(self.numAgent):
            agent = CleaningAgent(self)
            self.grid.place_agent(agent, pos=(1, 1))

        self.datacollector = DataCollector(
            model_reporters={"DirtyCells": lambda m: 1.0 - np.mean(m.dirtyGrid)},
            agent_reporters={"Movements": lambda a: a.movements}
        )

    def step(self):
        self.datacollector.collect(self)
        if self.actualWeight >= self.maxRounds or (1.0 - np.mean(self.dirtyGrid)) == 1.0:
            self.running = False
        else:
            self.agents.shuffle_do("step")
            self.actualWeight += 1

if __name__ == "__main__":
    m, n = 10, 10
    numAgent = 3
    dirtyPercentage = 0.5
    maxTime = 100

    print(f"Starting simulation with parameters: {m}x{n} grid, {numAgent} agents, {dirtyPercentage*100}% dirty cells, max time {maxTime} steps.")

    model = RoomToClean(m, n, numAgent, dirtyPercentage, maxTime)
    model.run_model()

    requiredTime = model.actualWeight
    print(f"\nSimulation finished in {requiredTime} steps.")

    finalPercentageCleaned = (1.0 - np.mean(model.dirtyGrid)) * 100
    print(f"Final percentage of cleaned cells: {finalPercentageCleaned:.2f}%")

    dataAgent = model.datacollector.get_agent_vars_dataframe()
    last_step_index = requiredTime - 1
    finalMovements = dataAgent.xs(last_step_index, level="Step")["Movements"].sum() if last_step_index >= 0 else 0
    print(f"Total movements made by all agents: {finalMovements}")
