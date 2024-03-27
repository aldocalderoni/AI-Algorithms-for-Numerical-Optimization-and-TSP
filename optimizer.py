from problem import *
from setup import *
import random
import math

class Optimizer(Setup):
    def __init__(self):
        Setup.__init__(self)
        self._pType = 0
        self._numExp = 0

    def setVariables(self,parameters):
        Setup.setVariables(self,parameters)
        self._pType = parameters['pType']
        self._numExp = parameters['numExp']

    def getNumExp(self):
        return self._numExp

    def displayNumExp(self):
        print()
        print("Number of experiments:",self._numExp)

    def displaySetting(self):
        pass

class HillClimbing(Optimizer):
    def __init__(self):
        super().__init__()
        self._numRestart = 0
        self._limitStuck = 0

    def setVariables(self,parameters):
        super().setVariables(parameters)
        self._numRestart = parameters['numRestart']
        self._limitStuck = parameters['limitStuck']

    def displaySetting(self):
        print()
        print("Number of random restarts: ", self._numRestart)
        if self._pType == 1:
            print()
            print("Mutation step size:",self._delta)

    def randomRestart(self, p):
        self.run(p)
        bestSolution = p.getSolution()
        bestValue = p.getValue()
        i=0
        while i<self._numRestart:
            self.run(p)
            newSolution = p.getSolution()
            newValue = p.getValue()
            if newValue<bestValue:
                bestSolution = newSolution
                newValue = bestValue
            i+=1
        p.storeResult(bestSolution,bestValue)

    def displayLimitStuck(self):
        print("Max evaluations with no improvement: {0:,} iterations".format(self._limitStuck))

    def run(self, p):
        pass

class SteepestAscent(HillClimbing):
    def __init__(self):
        super().__init__()

    def displaySetting(self):
        print()
        print("Search algorithm: Steepest-Ascent Hill Climbing")
        HillClimbing.displaySetting(self)

    def run(self,p):
        current = p.randomInit()
        valueC = p.evaluate(current)
        f = open('steepest.txt','w')
        while True:
            neighbors = p.mutants(current)
            successor, valueS = self.bestOf(neighbors,p)
            f.write(str(round(valueC,1))+'\n')
            if valueS >= valueC:
                break
            else:
                current = successor
                valueC = valueS
        f.close()
        p.storeResult(current, valueC)

    def bestOf(self, neighbors, p):
        best = neighbors[0]
        bestValue = p.evaluate(best)
        for i in range(1, len(neighbors)):
            tmp = p.evaluate(neighbors[i])
            if bestValue > tmp:
                best = neighbors[i]
                bestValue = tmp
        return best, bestValue

class FirstChoice(HillClimbing):
    def __init__(self):
        super().__init__()

    def displaySetting(self):
        print()
        print("Search algorithm: First-Choice Hill Climbing")
        HillClimbing.displaySetting(self)
        super().displayLimitStuck()

    def run(self, p):
        current = p.randomInit()
        valueC = p.evaluate(current)
        i = 0
        f = open('first.txt', 'w')
        while i < self._limitStuck:
            successor = p.randomMutant(current)
            valueS = p.evaluate(successor)
            f.write(str(round(valueC, 1)) + '\n')
            if valueS < valueC:
                current = successor
                valueC = valueS
                i = 0
            else:
                i += 1
        f.close()
        p.storeResult(current, valueC)

class Stochastic(HillClimbing):
    def displaySetting(self):
        print()
        print("Search algorithm: Stochastic Hill Climbing")
        HillClimbing.displaySetting(self)
        super().displayLimitStuck()

    def run(self,p):
        current = p.randomInit()
        valueC = p.evaluate(current)
        f = open('stochastic.txt','w')
        i=0
        while i<self._limitStuck:
            neighbors = p.mutants(current)
            successor, valueS = self.stochasticBest(neighbors,p)
            f.write(str(round(valueC,1))+'\n')
            if valueS < valueC:
                current = successor
                valueC = valueS
                i = 0
            else:
                i += 1
        f.close()
        p.storeResult(current, valueC)

    def stochasticBest(self, neighbors, p):
        # Smaller valuse are better in the following list
        valuesForMin = [p.evaluate(indiv) for indiv in neighbors]
        #print("valueformin",valuesForMin)
        largeValue = max(valuesForMin) + 1
        #print(largeValue)
        valuesForMax = [largeValue - val for val in valuesForMin]
        #print(valuesForMax)
        # Now, larger values are better
        total = sum(valuesForMax)
        #print(total)
        randValue = random.uniform(0, total)
        #print(randValue)
        s = valuesForMax[0]
        for i in range(len(valuesForMax)):
            #print(s)
            if randValue <= s:  # The one with index i is chosen
                break
            else:
                s += valuesForMax[i + 1]
        return neighbors[i], valuesForMin[i]

class GradientDescent(HillClimbing):
    def displaySetting(self):
        print()
        print("Search algorithm: Gradient Descent")
        print()
        print("Update rate:", self._alpha)
        print("Increment for calculating derivative:", self._dx)

    def run(self, p):
        currentP = p.randomInit()
        valueC = p.evaluate(currentP)
        f = open('gradient_descent.txt', 'w')
        while True:
            nextP = p.takeStep(currentP, valueC)
            valueN = p.evaluate(nextP)
            f.write(str(round(valueC, 1)) + '\n')
            if valueN >= valueC:
                break
            else:
                currentP = nextP
                valueC = valueN
        f.close()
        p.storeResult(currentP, valueC)

class MetaHeuristics(Optimizer):
    def __init__(self):
        super().__init__()

    def setVariables(self,parameters):
        super().setVariables(parameters)

class SimulatedAnnealing(MetaHeuristics):
    def __init__(self):
        super().__init__()
        self._limitEval = 0 # 몇 번 실행할 것인지
        self._sumOfWhen = 0 # When the best solution is found
        self._numSample = 100

    def setVariables(self,parameters):
        super().setVariables(parameters)
        self._limitEval  = parameters['limitEval']

    def initTemp(self, p):  # To set initial acceptance probability to 0.5
        diffs = []
        for i in range(self._numSample):
            c0 = p.randomInit()  # A random point
            v0 = p.evaluate(c0)  # Its value
            c1 = p.randomMutant(c0)  # A mutant
            v1 = p.evaluate(c1)  # Its value
            diffs.append(abs(v1 - v0))
        dE = sum(diffs) / self._numSample   # Average value difference
        t = dE / math.log(2)  # exp(–dE/t) = 0.5
        return t

    def tSchedule(self, t):
        return t * (1 - (1 / 10**4))

    def getWhenBestFound(self):
        return self._sumOfWhen

    def run(self, p):
        current = p.randomInit()
        valueC = p.evaluate(current)
        t = self.initTemp(p)
        f = open('anneal.txt', 'w')
        best,valueBset = current,valueC
        whenBestFound = i = 1
        while t == 0 or i<self._limitEval:
            t = self.tSchedule(t)
            successor = p.randomMutant(current)
            valueS = p.evaluate(successor)
            f.write(str(round(valueC, 1)) + '\n')
            dE = valueS - valueC
            if dE<0 or random.uniform(0,1) < math.exp(-dE/t):
                current = successor
                valueC = valueS
            if valueC< valueBset:
                (best,valueBset) = (current,valueC)
                whenBestFound = i
            i+=1
        self._sumOfWhen = whenBestFound
        f.close()
        p.storeResult(current, valueC)
        
    def displaySetting(self):
        print()
        print("Search algorithm: SimulatedAnnealing")

class GA(MetaHeuristics):
    def __init__(self):
        MetaHeuristics.__init__(self)
        self._popSize = 0     # Population size
        self._uXp = 0   # Probability of swappping a locus for Xover
        self._mrF = 0   # Multiplication factor to 1/n for bit-flip mutation
        self._XR = 0    # Crossover rate for permutation code
        self._mR = 0    # Mutation rate for permutation code
        self._pC = 0    # Probability parameter for Xover
        self._pM = 0    # Probability parameter for mutation

    def setVariables(self, parameters):
        MetaHeuristics.setVariables(self, parameters)
        self._popSize = parameters['popSize']
        self._uXp = parameters['uXp']
        self._mrF = parameters['mrF']
        self._XR = parameters['XR']
        self._mR = parameters['mR']
        if self._pType == 1:
            self._pC = self._uXp
            self._pM = self._mrF
        if self._pType == 2:
            self._pC = self._XR
            self._pM = self._mR

    def displaySetting(self):
        print()
        print("Search Algorithm: Genetic Algorithm")
        MetaHeuristics.displaySetting(self)
        print()
        print("Population size:", self._popSize)
        if self._pType == 1:   # Numerical optimization
            print("Number of bits for binary encoding:", self._resolution)
            print("Swap probability for uniform crossover:", self._uXp)
            print("Multiplication factor to 1/L for bit-flip mutation:",
                  self._mrF)
        elif self._pType == 2: # TSP
            print("Crossover rate:", self._XR)
            print("Mutation rate:", self._mR)
            
    def evalAndFindBest(self, pop, p):
        bestInd = None
        bestFitness = float('inf')

        for ind in pop:
            p.evalInd(ind)
            fitness = ind[0]
            if fitness < bestFitness:
                bestFitness = fitness
                bestInd = ind

        return bestInd
    
    def selectParents(self, pop):
        parent1 = self.binaryTournament(pop)
        parent2 = self.binaryTournament(pop)
        return parent1, parent2
    
    def selectTwo(self, pop):
        return random.choice(pop), random.choice(pop)
    
    def binaryTournament(self, pop):
        # Select the winner between two individuals
        ind1, ind2 = self.selectTwo(pop)
        if ind1[0] < ind2[0]:  # Assuming fitness is stored at index 0 of an individual
            return ind1
        else:
            return ind2
        
    def selectNewPop(self, oldPop, p):
        elitism_rate = 0.05
        num_elites = int(elitism_rate * len(oldPop))
        
        if((num_elites % 2 != 0 and len(oldPop) % 2 == 0) or (num_elites % 2 == 0 and len(oldPop) % 2 != 0)):
            num_elites -= 1

        oldPop.sort(key=lambda x: x[0])
        elites = oldPop[:num_elites]
        newPop = elites[:]
        
        while len(newPop) < len(oldPop):
            parent1, parent2 = self.selectParents(oldPop)
            child1, child2 = p.crossover(parent1, parent2, self._pC)
            child1 = p.mutation(child1, self._pM)
            child2 = p.mutation(child2, self._pM)
            newPop.extend([child1, child2])
        
        newPop.extend(elites)
        return newPop
    
    def run(self, p):
        num_generations = 50
        pop = p.initializePop(self._popSize)
        bestInd = self.evalAndFindBest(pop, p)
        f = open('genetic_algorithm.txt', 'w')
        
        for generation in range(num_generations):
            pop = self.selectNewPop(pop, p)
            bestInd = self.evalAndFindBest(pop, p)
            valueC = bestInd[0]
            f.write(str(round(valueC, 1)) + '\n')

        bestSolution = p.indToSol(bestInd)
        bestFitness = bestInd[0]
        f.close()
        p.storeResult(bestSolution, bestFitness)
    