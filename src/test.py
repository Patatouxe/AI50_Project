#Fonctionne très bien sur les instances Augerat (Set A) de 10 à 20% d'éloignement pour un temps entre 10 et 70s
# Beaucoup moins sur les instances TAI, éloignement entre 30 et 45% pour un temps ~60s (75 nodes) 

from CVRP import CVRP
from ACO_Classic.Colony import Colony
import time

start = time.time()
cvrpInstance = CVRP('D:\IA50\AI50_Project\src\data\\tai\\tai75a.vrp')
colony = Colony(cvrpInstance,75,200,2,5,9,0.8,80,100)
result = colony.solve()
optimal = 1445
end = time.time()
i=0
for route in result[0]:
    i+=1
    print("Route #{} : {}".format(i,route))
print("Cost : {}".format(result[1]))
print("Time : {}".format(end-start))
print("RPD : {}".format(100*(result[1]-optimal)/optimal))