import random
import math
from setup import *

class Problem(Setup):
    def __init__(self):
        super().__init__()
        self._solution = []
        self._value = 0
        self._numEval = 0
        self._pFileName = ""
        self._bestSolution = []
        self._bestMinimum = 0
        self._avgMinimum = 0
        self._avgNumEval = 0
        self._sumOfNumEval = 0
        self._avgWhen = 0

    def setVariables(self, parameters):
        Setup.setVariables(self, parameters)
        self._pFileName = parameters['pFileName']

    def randomInit(self):
        pass

    def evaluate(self, current):
        pass

    def mutants(self, current):
        pass

    def randomMutant(self, current):
        pass

    def describe(self):
        pass

    def getSolution(self):
        return self._solution

    def getValue(self):
        return self._value

    def getNumEval(self):
        return self._numEval

    def setNumEval(self, num=0):
        self._numEval += num

    def storeResult(self, solution, value):
        self._solution = solution
        self._value = value

    def storeExpResult(self,results):
        self._bestSolution = results[0]
        self._bestMinimum = results[1]
        self._avgMinimum = results[2]
        self._avgNumEval = results[3]
        self._sumOfNumEval = results[4]
        self._avgWhen = results[5]

    def report(self):
        pass

    def reportAvg(self):
        print()
        print("Total number of evaluations: {0:,}".format(self._sumOfNumEval)) #HeeHee
        print()
        print("Average objective value: {0:,.3f}".format(self._avgMinimum))
        print("Average number of evaluations: {0:,}".format(self._avgNumEval))


class Numeric(Problem):
    def __init__(self):
        # to inherit self._delta, self._alpha, self._dx, and self._limit_stuck from Setup,
        # and self._solution, self._value, self._numEval from Problem.
        super().__init__()
        self._expression = ""
        self._domain = []

    # getter functions
    def getDelta(self):
        return self._delta

    def getDx(self):
        return self._dx

    def getAlpha(self):
        return self._alpha

    def setVariables(self, parameters):
        super().setVariables(parameters)
        infile = open(self._pFileName, 'r')
        expression = ""
        var = []
        low = []
        up = []
        for index, line in enumerate(infile):
            if index == 0:
                expression = line.strip()
            else:
                temp = line.strip().split(",")
                var.append(temp[0])
                low.append(float(temp[1]))
                up.append(float(temp[2]))
        domain = [var, low, up]
        infile.close()

        self._expression = expression
        self._domain = domain

    def randomInit(self):
        init = []
        num_vars = len(self._domain[0])
        low = self._domain[1]
        up = self._domain[2]
        for i in range(num_vars):
            init.append(random.randint(low[i], up[i]))
        return init

    def evaluate(self, current):
        self.setNumEval(1)
        expr = self._expression  # p[0] is function expression
        varNames = self._domain[0]  # p[1] is domain: [varNames, low, up]
        for i in range(len(varNames)):
            assignment = varNames[i] + '=' + str(current[i])
            exec(assignment)
        return eval(expr)

    def mutate(self, current, i, d):
        curCopy = current[:]
        domain = self._domain  # [VarNames, low, up]
        low_ith = domain[1][i]  # Lower bound of i-th
        up_ith = domain[2][i]  # Upper bound of i-th
        if low_ith <= (curCopy[i] + d) <= up_ith:
            curCopy[i] += d
        return curCopy

    def randomMutant(self, current):
        i = random.randrange(len(current))
        d = (-1) ** random.randrange(0, 2) * self._delta
        return self.mutate(current, i, d)

    def mutants(self, current):
        neighbors = []
        for i in range(len(current)):
            neighbors.append(self.mutate(current, i, self._delta))
            neighbors.append(self.mutate(current, i, -self._delta))
        return neighbors

    def takeStep(self, currentP, valueC):
        nextP = currentP[:]
        gradients = self.gradient(currentP, valueC)
        for i in range(len(currentP)):
            nextP_temp_ith = currentP[i] - self._alpha * gradients[i]
            if self.isLegal(currentP[:i] + [nextP_temp_ith] + currentP[i + 1:]):
                nextP[i] = nextP_temp_ith
            #  if not, nextP[i] = currentP[i] by default.
        return nextP

    def gradient(self, currentP, valueC):
        gradients = []
        for i in range(len(currentP)):
            gradient_ith = currentP[:]

            #  maybe not necessary but I thought it would help.
            d = (-1) ** random.randrange(0, 2) * self._dx

            gradient_ith[i] += d
            valueN_ith = self.evaluate(gradient_ith)

            gradient_ith[i] = (valueN_ith - valueC) / d
            gradients.append(gradient_ith[i])
        return gradients

    def isLegal(self, nextP):
        low = self._domain[1]
        up = self._domain[2]
        for i in range(len(nextP)):
            if not low[i] <= nextP[i] <= up[i]:
                return False
        return True

    def describe(self):
        print()
        print("Objective function:")
        print(self._expression)  # Expression
        print()
        print("Search space:")
        varNames = self._domain[0]  # p[1] is domain: [VarNames, low, up]
        low = self._domain[1]
        up = self._domain[2]
        for i in range(len(low)):
            print(" " + varNames[i] + ":", (low[i], up[i]))

    def report(self):
        super().reportAvg()
        print()
        print("Best solution found:")
        print(self.coordinate())  # Convert list to tuple
        print("Best value: {0:,.3f}".format(self._bestMinimum))
        super().report()

    def coordinate(self):
        c = [round(value, 3) for value in self._bestSolution]
        return tuple(c)
    
    def initializePop(self, size): # Make a population of given size
        pop = []
        for i in range(size):
            chromosome = self.randBinStr()
            pop.append([0, chromosome])
        return pop

    def randBinStr(self):
        k = len(self._domain[0]) * self._resolution
        chromosome = []
        for i in range(k):
            allele = random.randint(0, 1)
            chromosome.append(allele)
        return chromosome

    def evalInd(self, ind):  # ind: [fitness, chromosome]
        ind[0] = self.evaluate(self.decode(ind[1])) # Record fitness

    def decode(self, chromosome):
        r = self._resolution
        low = self._domain[1]  # list of lower bounds
        up = self._domain[2]   # list of upper bounds
        genotype = chromosome[:]
        phenotype = []
        start = 0
        end = r   # The following loop repeats for # variables
        for var in range(len(self._domain[0])): 
            value = self.binaryToDecimal(genotype[start:end],
                                         low[var], up[var])
            phenotype.append(value)
            start += r
            end += r
        return phenotype

    def binaryToDecimal(self, binCode, l, u):
        r = len(binCode)
        decimalValue = 0
        for i in range(r):
            decimalValue += binCode[i] * (2 ** (r - 1 - i))
        return l + (u - l) * decimalValue / 2 ** r

    def crossover(self, ind1, ind2, uXp):
        # pC is interpreted as uXp# (probability of swap)
        chr1, chr2 = self.uXover(ind1[1], ind2[1], uXp)
        return [0, chr1], [0, chr2]

    def uXover(self, chrInd1, chrInd2, uXp): # uniform crossover
        chr1 = chrInd1[:]  # Make copies
        chr2 = chrInd2[:]
        for i in range(len(chr1)): 
            if random.uniform(0, 1) < uXp:
                chr1[i], chr2[i] = chr2[i], chr1[i]
        return chr1, chr2

    def mutation(self, ind, mrF):  # bit-flip mutation
        # pM is interpreted as mrF (factor to adjust mutation rate)
        child = ind[:]    # Make copy
        n = len(ind[1])
        for i in range(n):
            if random.uniform(0, 1) < mrF * (1 / n):
                child[1][i] = 1 - child[1][i]
        return child

    def indToSol(self, ind):
        return self.decode(ind[1])

class TSP(Problem):
    def __init__(self):
        # to inherit self._delta, self._alpha, self._dx, and self._limit_stuck from Setup,
        # and self._solution, self._value, self._numEval from Problem.
        super().__init__()

        # member variables below are specific to TSP.
        self._numCities = 0
        self._locations = []
        self._table = []

    def setVariables(self, parameters):
        super().setVariables(parameters)
        infile = open(self._pFileName, 'r')
        # First line is number of cities
        self._numCities = int(infile.readline())
        line = infile.readline()  # The rest of the lines are locations
        while line != '':
            self._locations.append(eval(line))  # Make a tuple and append
            line = infile.readline()
        infile.close()
        self._table = self.calcDistanceTable()

    def calcDistanceTable(self):
        table = []
        for i in range(self._numCities):
            row = []
            for j in range(self._numCities):
                dx = self._locations[i][0] - self._locations[j][0]
                dy = self._locations[i][1] - self._locations[j][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                row.append(dist)
            table.append(row)
        return table

    def randomInit(self):
        n = self._numCities
        init = list(range(n))
        random.shuffle(init)
        return init

    def evaluate(self, current):
        self.setNumEval(1)
        cost = 0
        for i in range(-1, len(current) - 1):
            cost += self._table[current[i]][current[i + 1]]
        return cost

    def inversion(self, current, i, j):
        curCopy = current[:]
        while i < j:
            curCopy[i], curCopy[j] = curCopy[j], curCopy[i]
            i += 1
            j -= 1
        return curCopy

    def randomMutant(self, current):
        while True:
            i, j = sorted([random.randrange(self._numCities) for _ in range(2)])
            if i < j:
                curCopy = self.inversion(current, i, j)
                break
        return curCopy

    def mutants(self, current):
        neighbors = []
        count = 0
        triedPairs = []
        while count <= self._numCities:
            i, j = sorted([random.randrange(self._numCities) for _ in range(2)])
            if i < j and [i, j] not in triedPairs:
                triedPairs.append([i, j])
                curCopy = self.inversion(current, i, j)
                count += 1
                neighbors.append(curCopy)
        return neighbors

    def describe(self):
        print()
        print("Number of cities:", self._numCities)
        print()
        print("City locations:")
        for i in range(self._numCities):
            print("{0:>12}".format(str(self._locations[i])), end='')
            if i % 5 == 4:
                print()

    def report(self):
        super().reportAvg()
        print()
        print("Best solution found:")
        self.tenPerRow()
        print()
        print("Best value: {0:,}".format(round(self._bestMinimum)))
        super().report()

    def tenPerRow(self):
        for i in range(len(self._bestSolution)):
            print("{0:>5}".format(self._bestSolution[i]), end='')
            if i % 10 == 9:
                print()

    def initializePop(self, size): # Make a population of given size
        n = self._numCities        # n: number of cities
        pop = []
        for i in range(size):
            chromosome = self.randomInit()
            pop.append([0, chromosome])
        return pop

    def evalInd(self, ind):  # ind: [fitness, chromosome]
        ind[0] = self.evaluate(ind[1]) # Record fitness

    def crossover(self, ind1, ind2, XR): 
        # pC is interpreted as XR (crossover rate)
        if random.uniform(0, 1) <= XR:
            chr1, chr2 = self.oXover(ind1[1], ind2[1])
        else:
            chr1, chr2 = ind1[1][:], ind2[1][:]  # No change
        return [0, chr1], [0, chr2]

    def oXover(self, chrInd1, chrInd2):  # Ordered Crossover
        chr1 = chrInd1[:]
        chr2 = chrInd2[:]  # Make copies
        size = len(chr1)
        a, b = sorted([random.randrange(size) for _ in range(2)])
        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:
                holes1[chr2[i]] = False
                holes2[chr1[i]] = False
        # We must keep the original values somewhere
        # before scrambling everything
        temp1, temp2 = chr1, chr2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                chr1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1
            if not holes2[temp2[(i + b + 1) % size]]:
                chr2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1
        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            chr1[i], chr2[i] = chr2[i], chr1[i]
        return chr1, chr2

    def mutation(self, ind, mR): # mutation by inversion
        # pM is interpreted as mR (mutation rate for inversion)
        child = ind[:]  # Make copy
        if random.uniform(0, 1) <= mR:
            i, j = sorted([random.randrange(self._numCities)
                           for _ in range(2)])
            child[1] = self.inversion(child[1], i, j)
        return child

    def indToSol(self, ind):
        return ind[1]