import logging
import os
import time
from src.CVRP import CVRP
from src.Saving_Algo.saving_class import Savings
from src.ACO_MACS.aco_colony import MACS_CVRP
from src.Gen_Algo.AG import GeneticAlgorithmCVRP

def main():
    """
    Main launch of CVRP solution evaluation using Savings, ACO Colony, and Genetic Algorithm.
    """
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("This is a test message to confirm the logger is working.")

    # Dynamically construct the path to test.txt
    project_root = os.path.dirname(os.path.abspath(__file__))
    instance_file_path = os.path.join(project_root, "test.txt")

    # Debugging path
    logger.info(f"Looking for instance file at: {instance_file_path}")

    try:
        # Read the instance file
        with open(instance_file_path, 'r') as file:
            instance_paths = [line.strip() for line in file if line.strip()]
            logger.info(f"Loaded instance paths: {instance_paths}")

        for instance_path in instance_paths:
            instance_path = instance_path.strip()
            logger.info(f"Processing instance: {instance_path}")

            # Initialize the CVRP instance
            cvrp_instance = CVRP(instance_path)
            logger.info(f"Loaded CVRP instance: {cvrp_instance.name}")
            logger.info(f"Number of customers: {cvrp_instance.dimension}")
            logger.info(f"Capacity per truck: {cvrp_instance.capacity}")
            """
            # Run the ACO Colony algorithm
            try:
                logger.info("Running ACO Colony algorithm...")
                start_time = time.time()
                macs_solver = MACS_CVRP(cvrp_instance)
                macs_solution, macs_distance, macs_vehicles = macs_solver.run(200)
                end_time = time.time()
                aco_execution_time = end_time - start_time

                # Validate the solution
                is_feasible, message = cvrp_instance.validate_solution(macs_solution)
                if not is_feasible:
                    logger.warning(f"ACO Colony solution invalid: {message}")

                # Log ACO Colony results
                formatted_solution = cvrp_instance.format_solution(macs_solution, macs_distance)
                logger.info("\nACO Colony Algorithm Results:")
                logger.info(formatted_solution)
                logger.info(f"Total distance: {macs_distance:.2f}")
                logger.info(f"Number of vehicles: {macs_vehicles}")
                logger.info(f"Average Distance per Route: {macs_distance / macs_vehicles:.2f}")
                logger.info(f"ACO Colony Execution Time: {aco_execution_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in ACO Colony algorithm: {str(e)}")
            """
            # Run the Genetic Algorithm
            try:
                logger.info("Running Genetic Algorithm...")
                start_time = time.time()
                ga_solver = GeneticAlgorithmCVRP(cvrp_instance, generations=200, population_size=100, mutation_rate=0.05)
                ga_solution, ga_total_distance = ga_solver.run()
                end_time = time.time()
                ga_execution_time = end_time - start_time

                # Validate the solution
                is_feasible, message = cvrp_instance.validate_solution(ga_solution)
                if not is_feasible:
                    logger.warning(f"Genetic Algorithm solution invalid: {message}")

                # Log Genetic Algorithm results
                logger.info("\nGenetic Algorithm Results:")
                formatted_solution = cvrp_instance.format_solution(ga_solution, ga_total_distance)
                logger.info(formatted_solution)
                logger.info(f"Genetic Algorithm Execution Time: {ga_execution_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in Genetic Algorithm: {str(e)}")
            """
            # Run the Savings algorithm
            try:
                logger.info("Running Savings algorithm...")
                start_time = time.time()
                savings_solver = Savings(cvrp_instance)
                savings_solution, savings_distance = savings_solver.run()
                end_time = time.time()
                savings_execution_time = end_time - start_time

                # Validate the solution
                is_feasible, message = cvrp_instance.validate_solution(savings_solution)
                if not is_feasible:
                    logger.warning(f"Savings solution invalid: {message}")

                # Log Savings results
                formatted_solution = cvrp_instance.format_solution(savings_solution, savings_distance)
                logger.info("\nSavings Algorithm Results:")
                logger.info(formatted_solution)
                logger.info(f"Total distance: {savings_distance:.2f}")
                logger.info(f"Number of routes: {len(savings_solution)}")
                logger.info(f"Average Distance per Route: {savings_distance / len(savings_solution):.2f}")
                logger.info(f"Savings Algorithm Execution Time: {savings_execution_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in Savings algorithm: {str(e)}")
            """
    except FileNotFoundError:
        logger.error(f"File not found: {instance_file_path}")
    except Exception as e:
        logger.error(f"Error in main execution occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
