# Main launch of CVRP solution evaluation

#Author : JosueKoanga

from src.CVRP import CVRP

#Import other  algorithm classes
from src.ACO.aco_colony import MACS_CVRP

def main():

    ## MACS- ACO algorithm

    #Initialize the CVRP instance with the path to the data file
    cvrp_macs_instance = CVRP("data/tai/tai150a.vrp")
    print(f' CVRP instance: {cvrp_macs_instance.name} with {cvrp_macs_instance.dimension} customers')
    print(f' Capacity per truck: {cvrp_macs_instance.capacity}')

    #Initialize the MACS_CVRP instance with the CVRP instance
    macs = MACS_CVRP(cvrp_macs_instance, 0.1, 0.1)
    print(f' MACS instance Initialized...')

    #Print the best solution found by the MACS algorithm
    print(f' Running MACS algorithm...')
    macs_solution, macs_distance, macs_vehicles = macs.run(100)

    print(f"MACS solution found by MACS: {macs_solution}")
    print(f"MACS total distance: {macs_distance}")
    print(f"MACS number of vehicles: {macs_vehicles}")

    ## Other Algorithms

    ##Compare results

if __name__ == "__main__":
    main()