from CVRP import CVRP
from ACO_Classic.Colony import Colony
import time

start = time.time()
cvrpInstance = CVRP('D:\IA50\AI50_Project\src\data\\tai\\tai75b.vrp')
colony = Colony(cvrpInstance,75,200,2,5,9,0.8,80,100)
result = colony.solve()

end = time.time()
i=0
for route in result[0]:
    i+=1
    print("Route #{} : {}".format(i,route))
print("Cost : {}".format(result[1]))
print("Time : {}".format(end-start))
print("RPD : {}".format((result[1]-1344.62)/1344.62))