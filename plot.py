import matplotlib.pylab as pl

def main():
    pl.figure()
    pl.title("Search Performance (TSP-100)")
    pl.xlabel("Number of Evaluations")
    pl.ylabel("Tour Cost")

    f = open("first.txt", 'r') 
    listPoint = [float(line.rstrip()) for line in f]
    pl.plot(listPoint,label = "First-Choice Hill Climbing")
    f.close()

    f = open("anneal.txt", 'r')
    listPoint = [float(line.rstrip()) for line in f]
    pl.plot(listPoint, label = "Simulated annealing")
    f.close()

    pl.legend()
    pl.show()

main()