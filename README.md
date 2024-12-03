
# AI50 Project - Capacitated Vehicle Routing Problem (CVRP)

An implementation of Multiple Ant Colony System for solving the Capacitated Vehicle Routing Problem (CVRP).

## Requirements

- Python 3.10+
- Dependencies:
  - numpy
  - matplotlib
  - tqdm
  - requests_toolbelt

## Installation

```bash
# Clone the repository
git clone [https://github.com/Patatouxe/AI50_Project.git]

# Install dependencies
pip install numpy matplotlib tqdm requests_toolbelt
```

## Project Structure

```
.
├── data/
│   └── tai/           # CVRP benchmark instances
├── src/
│   ├── ACO/
│   │   └── aco_colony.py  # MACS-CVRP implementation
│   └── CVRP.py       # CVRP problem definition
└── main.py           # Main execution script
```

## Usage

Run the main script to execute the CVRP whole system:

```bash
python main.py
```

The script will:
1. Load a CVRP instance from the data directory
2. Initialize the MACS-CVRP solver
3. Run the optimization for 100 iterations
4. Display the best solution found, including:
   - Route assignments
   - Total distance
   - Number of vehicles used

## Algorithm Details

We can find below a formal description for the CVRP:

- **Objective:** The objective is to minimize the vehicle fleet and the sum of distance, and the total demand of commodities for each route may not exceed the capacity of the vehicle which serves that route.

- **Feasibility:** A solution is feasible if the total quantity assigned to each route does not exceed the capacity of the vehicle which services the route.

- **Formulation:** Let Q denote the capacity of a vehicle. Mathematically, a solution for the CVRP is the same that VRP's one, but with the additional restriction that the total demand of all customers supplied on a route $R_{i}$ does not exceed the vehicle capacity

  	 $$Q: \sum_{i=1}^{m} q_{i} \leq Q$$


#### Variables: 
-  $n_{customer}$ : number of customers to be served n a unique depot
-  $q_{i}$ : Quantity of goods demanded by the customer (i = 1, ..,n)
-  $Q_{vehicle}$ : Capacity of vehicle available to deliver the good. Vehicle has to return to reload.
- $R_{i}$ : Route used by a vehicle to supply customer
-  $d_{ij}$ : Distance from customer i to customer j
#### Solution: 
Our solution of CVRP will be represented by a collection(array) of tours on the routes where all customers are served, and the total tour demand is Q

Objectif function : 

			$$S_{x} = [[(R_{i}, T_{i}), \sum_{i=1} q_{i}];    [(R_{n}, T_{n}), \sum_{n=1} q_{n}]] 
					where Q = max(total q)$$
Graphical theoritical formulation, 
	For a graph  G = (C, L),
		we have $c_{0}$ as the depot and $c_{n}/{n≠0}$ each note has a fixed demanded quantity  $q_{n}$. With $G_{0}= (c_{0},(c_{i},c_{j}) i≠j≠0)$ and $q_{0} = 0$  
		Each arc $(c_{i},c_{j})$ has a distance $d_{ij}$


#### MACS-CVRP (Multi Ant Colony System in Capacitated Vehicle Routing Problem)
The implementation uses Multiple Ant Colony System (MACS) with two colonies:
- ACS-VEI: Minimizes the number of vehicles
- ACS-DIST: Minimizes the total travel distance

Key parameters:
- `rho`: Global pheromone evaporation rate (default: 0.1)
- `local_rho`: Local pheromone update rate (default: 0.1)

## Benchmarks

The project uses Taillard's CVRP benchmark instances. Solutions can be compared against:
- Rochat and Taillard, 1995
- Potvin and Bengio, 1996 (PB)

## Development

### Testing
Add unit tests in a `tests/` directory (to be implemented).

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## License
[Add your license information here]
