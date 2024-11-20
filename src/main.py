# Main launch of CVRP solution evaluation

#Author : JosueKoanga
#Import other  algorithm classes
import logging
from pyexpat.errors import messages

from src.CVRP import CVRP
from src.ACO.aco_colony import MACS_CVRP

def main():
    #Set up the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    ## MACS- ACO algorithm
    try:

        #Initialize the CVRP instance with the path to the data file
        cvrp_macs_instance = CVRP("data/tai/tai150a.vrp")
        #print(f' CVRP instance: {cvrp_macs_instance.name} with {cvrp_macs_instance.dimension} customers')
        #print(f' Capacity per truck: {cvrp_macs_instance.capacity}')
        logger.info(f'Loaded CVRP instance: {cvrp_macs_instance.name}')
        logger.info(f'Number of customers: {cvrp_macs_instance.dimension}')
        logger.info(f'Capacity per truck: {cvrp_macs_instance.capacity}')

        #Initialize the MACS_CVRP instance with the CVRP instance
        macs = MACS_CVRP(cvrp_macs_instance)
        print(f' MACS instance Initialized...')

        #Print the best solution found by the MACS algorithm
        print(f' Running MACS algorithm...')
        macs_solution, macs_distance, macs_vehicles = macs.run(100)

        #validate macs_solution
        is_feasible, message = cvrp_macs_instance.validate_solution(macs_solution)
        if not is_feasible:
            logger.info(f"Invalid solution is not feasible: {message}")

        #Format and Display solution
        formatted_solution =  cvrp_macs_instance.format_solution(macs_solution, macs_distance)
        logger.info(f"\n Final Solution")
        logger.info(formatted_solution)

        logger.info(f' \n Solution statistics :')
        logger.info(f"Total distance: {macs_distance:.2f}")
        logger.info(f"Number of vehicles: {macs_vehicles}")
        logger.info(f"Average Distance per Route: {macs_distance/macs_vehicles:.2f}")

        #Visualize the solution
        macs.visualize_solution()

    except Exception as e:
        logger.error(f"Error in main execution occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()