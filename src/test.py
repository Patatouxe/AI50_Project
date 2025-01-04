#Fonctionne très bien sur les instances Augerat (Set A) de 10 à 20% d'éloignement pour un temps entre 10 et 70s
# Beaucoup moins sur les instances TAI, éloignement entre 30 et 45% pour un temps ~60s (75 nodes) 

from CVRP import CVRP
from ACO_Classic.Colony import Colony
import time

def launchColony():
    start = time.time()
    cvrpInstance = CVRP('C:/Users/guill/Documents/GitHub/AI50_Project/src/data/A/A-n63-k10.vrp')
    cvrpInstance.dimension
    colony = Colony(cvrpInstance,750,1,2,9,0.5,80,100)
    result = colony.solve()
    optimal = 1314
    end = time.time()
    i=0
    """ for route in result[0]:
        i+=1
        print("Route #{} : {}".format(i,route))
    print("Cost : {}".format(result[1])) """
    print("Time : {}".format(end-start))
    print("RPD : {}".format(100*(result[1]-optimal)/optimal))
    return end-start,100*(result[1]-optimal)/optimal

timeArr, rpdArr = [], [] 
for _ in range(5):
    tempTime, tempRpd = launchColony()
    timeArr.append(tempTime)
    rpdArr.append(tempRpd)
    
print("Average Time : {}".format(sum(timeArr)/len(timeArr)))
print("Average RPD : {}".format(sum(rpdArr)/len(rpdArr)))