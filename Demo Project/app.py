import numpy
import math
import random
import heapq
from flask import Flask, jsonify, request
from flask_cors import CORS # To allow cross-origin requests from your frontend

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# -- Bins and Grid Requirements --
num_bins = 50
grid_range = 15 # 15x15 km grid
bin_volume_min = 360 # kg
bin_volume_max = 660 # kg
bin_service_time = 0.1 # hours, constant for all bins for simplicity

bins_data_global = {} # Store this globally or pass around
totalTrash_global = 0

# -- Definitions for the Trucks --
truck_capacity = 10000 # kg
truck_speed = 40  # units/hour (remains for time calculation, though time cost is zeroed)
num_trucks_for_ga_start = 1 # Fixed for this problem as per requirement

start_depot_location = (0, 0)
end_depot_location = (0, 0) # Assuming trucks return to the same depot

# -- Define Incinerator --
incinerator_location = (30, 30) # In the top right corner away from the rest of the bins to simulate real-world conditions
incinerator_unload_time = 0.5  # hours
incinerator_id = 'INC'
depot_id = 'depot'

# -- Cost Parameters (prices in SGD) --
COST_PER_KM = 3.2382  # Per km (based on average km/L of Trash Trucks and price per L of diesel)
GATE_FEE = 80  # Per trip to incinerator
COST_PER_ADDITIONAL_TRUCK = 200 # Assumed Salary
MST_ERROR = (num_bins + 2) # Total Number of Nodes in the Graph

# -- GA Parameters --
sol_per_pop = 500
num_parents_mating = 128
num_generations = 1500
mutation_rate = 0.25
num_elite = 5 # Number of top individuals to carry over

# Fitness weights now primarily for penalties, as monetary costs are explicit
fitness_weights_simplified = {
    'unvisited': 1000000,            # High penalty for not visiting all bins
    'capacity_violation': 50000,     # High penalty for exceeding truck capacity
    'invalid_bin': 200000            # High penalty for invalid bin references
}

# Global variables to store GA results for specific generations
# We will store the full route and its associated metrics
generation_results = {}
theoretical_min_cost_benchmark = 0
fitness_history = [] # NEW: To store best fitness per generation for live plotting

# Utility Functions (keep these as they are, no changes needed for Flask integration itself)
def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two locations."""
    return numpy.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def get_location_coords(stop_id, bins_data_dict, depot_loc, incinerator_loc_coords,
                        depot_id_str, incinerator_id_str):
    """Returns the (x, y) coordinates for a given stop ID."""
    if stop_id == depot_id_str:
        return depot_loc
    elif stop_id == incinerator_id_str:
        return incinerator_loc_coords
    elif stop_id in bins_data_dict:
        return bins_data_dict[stop_id]['loc']
    else:
        return None

def calculate_mst_for_bins_and_depot(bins_data, depot_loc):
    """
    Calculates the cost of a Minimum Spanning Tree (MST) connecting only the depot
    and all bins. This provides a lower bound for the collection part of the routes.
    """
    nodes = {'depot': depot_loc}
    for bin_id, data in bins_data.items():
        nodes[bin_id] = data['loc']

    if not nodes:
        return 0

    min_cost = 0
    start_node_id = list(nodes.keys())[0]

    min_heap = [(0, start_node_id)] # (cost, node_id)
    visited = set()

    while min_heap and len(visited) < len(nodes):
        cost, current_node_id = heapq.heappop(min_heap)

        if current_node_id in visited:
            continue

        visited.add(current_node_id)
        min_cost += cost

        current_coords = nodes[current_node_id]

        for neighbor_id, neighbor_coords in nodes.items():
            if neighbor_id not in visited:
                distance = calculate_distance(current_coords, neighbor_coords)
                heapq.heappush(min_heap, (distance, neighbor_id))

    if len(visited) < len(nodes):
        return float('inf')

    return min_cost

def calculate_closest_bin_distace_theoretical(bins_data, incinerator_loc, depot_loc, min_trips, cost_per_km):
    theoretical_closest = float('inf')
    # Find the bin closest to the incinerator
    for bin_id, data in bins_data.items():
        dist = calculate_distance(data['loc'], incinerator_loc)
        if dist < theoretical_closest:
            theoretical_closest = dist

    return theoretical_closest

def calculate_incinerator_round_trip_cost(bins_data, incinerator_loc, depot_loc, min_trips, cost_per_km):
    """
    Calculates the theoretical minimum driving cost for incinerator trips.
    For each trip, it assumes travel from the bin closest to the incinerator,
    then to the incinerator, and then back to the depot.
    """
    if min_trips == 0 or not bins_data:
        return 0

    inc_to_depot = calculate_distance(incinerator_loc, depot_loc)
    closest_bin_to_inc_dist = calculate_closest_bin_distace_theoretical(bins_data, incinerator_loc, depot_loc, min_trips, cost_per_km)

    # Total distance for one theoretical incinerator trip
    distance_per_trip = 2*closest_bin_to_inc_dist
    final_distance = ((min_trips-1) * distance_per_trip) + closest_bin_to_inc_dist + inc_to_depot
    return final_distance * COST_PER_KM

# Core GA Functions (keep these as they are, but they will be called by Flask routes)
def calculate_fitness_vrp_simplified(chromosome, bins_data, truck_capacity, incinerator_location_coords,
                                     start_depot_location, end_depot_location, truck_speed,
                                     incinerator_unload_time, cost_per_km, gate_fee, cost_per_additional_truck, weights):
    """
    Calculates the fitness of a single VRP chromosome based on the new cost requirements.
    Capacity violation penalty and unvisited bin penalty are enforced.
    Mandatory final incinerator trip logic is enforced.
    """
    total_distance = 0
    total_time = 0 # Not directly used for fitness cost, but calculated
    total_incinerator_trips = 0 # Counts all incinerator trips, explicit and mandatory final

    trucks_used_in_chromosome = 0

    all_bins_in_problem_local = set(bins_data.keys())
    visited_bins_in_chromosome = set()

    capacity_violation_penalty = 0
    unvisited_bin_penalty = 0
    invalid_bin_penalty_val = 0

    for truck_idx, truck_route in enumerate(chromosome):
        current_load = 0
        current_truck_distance = 0
        current_truck_time = 0
        last_location = start_depot_location
        has_collected_any_bins_this_truck = False

        has_any_bins_in_route = any(stop_id in bins_data for stop_id in truck_route)
        if has_any_bins_in_route:
            trucks_used_in_chromosome = 1 # Fixed to 1 for this context as num_trucks_for_ga_start is 1

        if not truck_route or (len(truck_route) == 2 and truck_route[0] == depot_id and truck_route[1] == depot_id):
            continue

        processed_route = list(truck_route)
        if not processed_route or processed_route[0] != depot_id:
            processed_route.insert(0, depot_id)

        for i, stop_id in enumerate(processed_route):
            current_location = None

            if stop_id == depot_id:
                if i == 0:
                    current_location = start_depot_location
                else:
                    current_location = end_depot_location
            elif stop_id == incinerator_id:
                current_location = incinerator_location_coords
                total_incinerator_trips += 1
                current_truck_time += incinerator_unload_time
                current_load = 0 # Truck unloads at incinerator
            else:
                if stop_id not in bins_data:
                    invalid_bin_penalty_val += weights.get('invalid_bin', 200000)
                    continue
                bin_info = bins_data[stop_id]
                current_location = bin_info['loc']
                visited_bins_in_chromosome.add(stop_id)
                has_collected_any_bins_this_truck = True
                if current_load + bin_info['volume'] > truck_capacity:
                    capacity_violation_penalty += weights.get('capacity_violation', 50000) * \
                                                 (current_load + bin_info['volume'] - truck_capacity)
                current_load += bin_info['volume']
                current_truck_time += bin_info['service_time']

            if current_location is not None:
                dist = calculate_distance(last_location, current_location)
                current_truck_distance += dist
                current_truck_time += dist / truck_speed if truck_speed > 0 else float('inf')
                last_location = current_location

        # Mandatory final trip to incinerator if truck has collected trash and isn't empty
        if has_collected_any_bins_this_truck and current_load > 0:
            dist_to_inc = calculate_distance(last_location, incinerator_location_coords)
            current_truck_distance += dist_to_inc
            current_truck_time += (dist_to_inc / truck_speed if truck_speed > 0 else float('inf'))
            current_truck_time += incinerator_unload_time
            total_incinerator_trips += 1
            dist_inc_to_depot = calculate_distance(incinerator_location_coords, end_depot_location)
            current_truck_distance += dist_inc_to_depot
            current_truck_time += (dist_inc_to_depot / truck_speed if truck_speed > 0 else float('inf'))
        else: # Ensure truck returns to depot if it didn't go to incinerator as final step
            if last_location != end_depot_location:
                dist_to_final_depot = calculate_distance(last_location, end_depot_location)
                current_truck_distance += dist_to_final_depot
                current_truck_time += (dist_to_final_depot / truck_speed if truck_speed > 0 else float('inf'))

        total_distance += current_truck_distance
        total_time += current_truck_time

    unvisited_bins = all_bins_in_problem_local - visited_bins_in_chromosome
    if unvisited_bins:
        unvisited_bin_penalty = len(unvisited_bins) * weights.get('unvisited', 1000000)

    # Calculate monetary costs
    driving_cost = total_distance * cost_per_km
    gate_fees_cost = total_incinerator_trips * gate_fee

    additional_truck_cost = max(0, trucks_used_in_chromosome) * cost_per_additional_truck

    fitness = (driving_cost +
               gate_fees_cost +
               additional_truck_cost +
               unvisited_bin_penalty +
               capacity_violation_penalty +
               invalid_bin_penalty_val)

    return fitness

def create_initial_vrp_population(sol_per_pop, bins_data_keys, num_trucks, depot_id, incinerator_id, truck_capacity, bins_data):
    population = []
    all_bin_ids = list(bins_data_keys)
    for _ in range(sol_per_pop):
        chromosome = []
        shuffled_bins = list(all_bin_ids)
        random.shuffle(shuffled_bins)

        bins_per_truck_base = len(shuffled_bins) // num_trucks
        remainder_bins = len(shuffled_bins) % num_trucks

        bin_idx_start = 0
        for i in range(num_trucks):
            route = [depot_id]
            current_cap_sim = 0

            num_bins_for_this_truck = bins_per_truck_base
            if i < remainder_bins:
                num_bins_for_this_truck += 1

            bins_for_this_route = shuffled_bins[bin_idx_start : bin_idx_start + num_bins_for_this_truck]
            bin_idx_start += num_bins_for_this_truck

            for bin_id in bins_for_this_route:
                if bin_id in bins_data:
                    bin_volume = bins_data[bin_id]['volume']
                    if current_cap_sim + bin_volume > truck_capacity and current_cap_sim > 0:
                        route.append(incinerator_id)
                        current_cap_sim = 0
                    route.append(bin_id)
                    current_cap_sim += bin_volume

            if current_cap_sim > 0 and any(s in bins_data for s in route):
                 if route[-1] != incinerator_id:
                     route.append(incinerator_id)

            if route[-1] != depot_id:
                route.append(depot_id)
            chromosome.append(route)
        population.append(chromosome)
    return population

def calculate_fitness_for_population(population, bins_data, truck_capacity, incinerator_loc,
                                     start_depot_loc, end_depot_loc, speed, unload_time,
                                     cost_per_km, gate_fee, cost_per_additional_truck, fitness_weights):
    fitness_scores = []
    for chromo in population:
        fitness_scores.append(calculate_fitness_vrp_simplified(
            chromo, bins_data, truck_capacity, incinerator_loc,
            start_depot_loc, end_depot_loc, speed,
            unload_time, cost_per_km, gate_fee, cost_per_additional_truck, fitness_weights
        ))
    return numpy.array(fitness_scores)

def select_mating_pool_vrp(population, fitness, num_parents):
    parents = []
    sorted_indices = numpy.argsort(fitness)
    for i in range(num_parents):
        parents.append(population[sorted_indices[i]])
    return parents

def crossover_vrp(parents, offspring_size_tuple, bins_data_keys, depot_id, incinerator_id):
    offspring = []
    num_offspring_needed = offspring_size_tuple[0]
    if not parents: return []
    for k in range(num_offspring_needed):
        idx1 = random.randrange(len(parents))
        idx2 = random.randrange(len(parents))
        parent1 = parents[idx1]
        parent2 = parents[idx2]

        num_routes = len(parent1) if parent1 else len(parent2)
        if num_routes == 0:
            offspring.append([[depot_id, depot_id]])
            continue

        child_routes_raw = [[] for _ in range(num_routes)]

        crossover_point_route_idx = random.randint(0, num_routes - 1)

        for r_idx in range(num_routes):
            if r_idx <= crossover_point_route_idx:
                child_routes_raw[r_idx] = list(parent1[r_idx]) if r_idx < len(parent1) else [depot_id, depot_id]
            else:
                child_routes_raw[r_idx] = list(parent2[r_idx]) if r_idx < len(parent2) else [depot_id, depot_id]

        child = []
        for route in child_routes_raw:
            if not route or route[0] != depot_id:
                route.insert(0, depot_id)
            if route[-1] != depot_id:
                route.append(depot_id)

            cleaned_route = [route[0]]
            for j in range(1, len(route)):
                if (route[j] == depot_id and cleaned_route[-1] == depot_id) or \
                   (route[j] == incinerator_id and cleaned_route[-1] == incinerator_id):
                        # Avoid consecutive depot or incinerator entries unless it's the start/end depot
                    if not (j > 0 and route[j] == depot_id and cleaned_route[-1] == depot_id) and \
                       not (j > 0 and route[j] == incinerator_id and cleaned_route[-1] == incinerator_id):
                        cleaned_route.append(route[j])
                else:
                    cleaned_route.append(route[j])
            child.append(cleaned_route)

        offspring.append(child)
    return offspring


def mutation_vrp(offspring_crossover, mutation_rate, bins_data_keys, depot_id, incinerator_id):
    mutated_offspring = []
    all_bins_set = set(bins_data_keys)
    for chromosome in offspring_crossover:
        mutated_chromosome = [list(route) for route in chromosome]

        # Mutation 1: Swap two bins within a random route
        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            truck_idx_to_mutate = random.randrange(len(mutated_chromosome))
            route_to_mutate = mutated_chromosome[truck_idx_to_mutate]

            bin_indices_in_route = [i for i, stop in enumerate(route_to_mutate)
                                    if stop != depot_id and stop != incinerator_id]
            if len(bin_indices_in_route) >= 2:
                idx1_map, idx2_map = random.sample(bin_indices_in_route, 2)
                route_to_mutate[idx1_map], route_to_mutate[idx2_map] = route_to_mutate[idx2_map], route_to_mutate[idx1_map]

        # Mutation 2: Move a bin from one route to another (or within the same route)
        if random.random() < mutation_rate:
            if not mutated_chromosome: continue

            candidate_bins_to_move = []
            for r_idx, route in enumerate(mutated_chromosome):
                for s_idx, stop in enumerate(route):
                    if stop in all_bins_set:
                        candidate_bins_to_move.append((r_idx, s_idx, stop))

            if candidate_bins_to_move:
                source_route_idx, source_stop_idx, bin_to_move = random.choice(candidate_bins_to_move)

                # Ensure source_stop_idx is valid before popping
                if 0 <= source_stop_idx < len(mutated_chromosome[source_route_idx]):
                    mutated_chromosome[source_route_idx].pop(source_stop_idx)

                target_truck_idx = random.randrange(len(mutated_chromosome))
                target_route = mutated_chromosome[target_truck_idx]

                insert_pos = random.randrange(1, max(2, len(target_route) - 1))
                target_route.insert(insert_pos, bin_to_move)

                # Re-clean the target route after insertion
                if not target_route or target_route[0] != depot_id:
                    target_route.insert(0, depot_id)
                if target_route[-1] != depot_id:
                    target_route.append(depot_id)

                cleaned_target_route = [target_route[0]]
                for j in range(1, len(target_route)):
                    if (target_route[j] == depot_id and cleaned_target_route[-1] == depot_id) or \
                       (target_route[j] == incinerator_id and cleaned_target_route[-1] == incinerator_id):
                        if not (j > 0 and target_route[j] == depot_id and cleaned_target_route[-1] == depot_id) and \
                           not (j > 0 and target_route[j] == incinerator_id and cleaned_target_route[-1] == incinerator_id):
                            cleaned_target_route.append(target_route[j])
                    else:
                        cleaned_target_route.append(target_route[j])
                mutated_chromosome[target_truck_idx] = cleaned_target_route


        # Mutation 3: Add/Remove an incinerator trip (simple heuristic)
        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            truck_idx = random.randrange(len(mutated_chromosome))
            route = mutated_chromosome[truck_idx]

            if random.random() < 0.5 and len(route) > 2:
                insert_pos = random.randrange(1, len(route) - 1)
                # Ensure we don't insert INC next to another INC or at depot boundaries
                if route[insert_pos] != incinerator_id and route[insert_pos-1] != incinerator_id and \
                   route[insert_pos] != depot_id and route[insert_pos-1] != depot_id:
                    route.insert(insert_pos, incinerator_id)
            else:
                inc_indices = [i for i, stop in enumerate(route) if stop == incinerator_id and i != 0 and i != len(route)-1]
                if inc_indices:
                    route.pop(random.choice(inc_indices))

        mutated_offspring.append(mutated_chromosome)
    return mutated_offspring

# Flask Routes
@app.route('/generate_bins', methods=['GET'])
def generate_bins():
    global bins_data_global, totalTrash_global, fitness_history # Include fitness_history here to reset it
    bins_data_global = {}
    for i in range(num_bins):
        bin_id = f'{i+1}'
        loc_x = random.uniform(0, grid_range)
        loc_y = random.uniform(0, grid_range)
        volume = random.uniform(bin_volume_min, bin_volume_max)
        bins_data_global[bin_id] = {'loc': (loc_x, loc_y), 'volume': volume, 'service_time': bin_service_time}

    totalTrash_global = sum(bin_info['volume'] for bin_info in bins_data_global.values())
    fitness_history = [] # Reset fitness history when new bins are generated

    # Prepare bin data for JSON response, converting tuples to lists
    bins_for_json = {
        bin_id: {
            'loc': list(data['loc']),
            'volume': data['volume'],
            'service_time': data['service_time']
        } for bin_id, data in bins_data_global.items()
    }
    return jsonify({
        'bins_data': bins_for_json,
        'depot_location': list(start_depot_location),
        'incinerator_location': list(incinerator_location)
    })

@app.route('/run_ga', methods=['POST'])
def run_ga():
    global generation_results, theoretical_min_cost_benchmark, bins_data_global, totalTrash_global, fitness_history

    if not bins_data_global:
        return jsonify({"error": "Bins data not generated. Please call /generate_bins first."}), 400

    generation_results = {} # Reset results for a new GA run
    fitness_history = [] # Reset fitness history at the start of a new GA run
    all_bins_in_problem = set(bins_data_global.keys())
    bin_ids_list = list(bins_data_global.keys())

    # Calculate Minimum Theoretical Cost (Benchmark)
    mst_bins_depot_distance = calculate_mst_for_bins_and_depot(bins_data_global, start_depot_location)
    min_incinerator_trips_theoretical = math.ceil(totalTrash_global / truck_capacity) if truck_capacity > 0 else 0
    theoretical_incinerator_travel_cost = calculate_incinerator_round_trip_cost(
        bins_data_global, incinerator_location, start_depot_location, min_incinerator_trips_theoretical, COST_PER_KM
    )
    min_driving_cost_benchmark = (mst_bins_depot_distance * COST_PER_KM) + theoretical_incinerator_travel_cost
    min_gate_fees_cost_benchmark = min_incinerator_trips_theoretical * GATE_FEE
    min_truck_cost_benchmark = COST_PER_ADDITIONAL_TRUCK if totalTrash_global > 0 else 0
    theoretical_min_cost_benchmark = min_driving_cost_benchmark + min_gate_fees_cost_benchmark + min_truck_cost_benchmark + MST_ERROR

    current_num_trucks = num_trucks_for_ga_start
    new_population = create_initial_vrp_population(sol_per_pop, bin_ids_list, current_num_trucks, depot_id, incinerator_id, truck_capacity, bins_data_global)

    overall_best_fitness = float('inf')
    best_route_overall = []

    generations_to_save = [100, 200, 300, 400, 500, 1000, 1500]

    for generation in range(num_generations):
        fitness_scores = calculate_fitness_for_population(
            new_population, bins_data_global, truck_capacity,
            incinerator_location, start_depot_location,
            end_depot_location, truck_speed,
            incinerator_unload_time, COST_PER_KM, GATE_FEE, COST_PER_ADDITIONAL_TRUCK, fitness_weights_simplified
        )

        current_best_fitness_in_gen = numpy.min(fitness_scores)

        # Update overall best if current generation has a better fitness
        if current_best_fitness_in_gen < overall_best_fitness:
            overall_best_fitness = current_best_fitness_in_gen
            best_idx_in_pop = numpy.argmin(fitness_scores)
            best_route_overall = new_population[best_idx_in_pop]
        
        fitness_history.append(float(overall_best_fitness)) # Store the overall best fitness for this generation

        # Save data for specific generations
        if (generation + 1) in generations_to_save:
            print(f"\n--- Attempting to save generation: {generation + 1} ---")
            print(f"Overall best fitness at generation {generation + 1}: {overall_best_fitness:.2f}")
            print(f"Best route overall at this point (before conversion): {best_route_overall}")
            print(f"Type of best_route_overall: {type(best_route_overall)}")
            if best_route_overall:
                print(f"Length of best_route_overall: {len(best_route_overall)}")
                if len(best_route_overall) > 0:
                    print(f"First truck route in best_route_overall: {best_route_overall[0]}")
            else:
                print("best_route_overall is empty or None.")


            # Recalculate the detailed metrics for the 'overall_best_fitness' route
            # This ensures the metrics align with the best route saved.
            calculated_total_distance = 0
            calculated_incinerator_trips = 0
            calculated_trucks_used = 0

            if best_route_overall:
                for truck_route in best_route_overall:
                    current_truck_distance_temp = 0
                    current_load_temp = 0
                    last_location_temp = start_depot_location
                    has_collected_any_bins_this_truck_temp = False

                    # Check if the truck route contains any bins to count it as "used"
                    if any(stop_id in bins_data_global for stop_id in truck_route):
                        calculated_trucks_used = 1 # Fixed to 1 truck for this problem

                    processed_route_temp = list(truck_route)
                    if not processed_route_temp or processed_route_temp[0] != depot_id:
                        processed_route_temp.insert(0, depot_id)

                    for i, stop_id in enumerate(processed_route_temp):
                        current_location_temp = get_location_coords(stop_id, bins_data_global, start_depot_location, incinerator_location, depot_id, incinerator_id)
                        if current_location_temp:
                            dist = calculate_distance(last_location_temp, current_location_temp)
                            current_truck_distance_temp += dist
                            last_location_temp = current_location_temp

                        if stop_id == incinerator_id:
                            calculated_incinerator_trips += 1
                            current_load_temp = 0
                        elif stop_id in bins_data_global:
                            current_load_temp += bins_data_global[stop_id]['volume']
                            has_collected_any_bins_this_truck_temp = True

                    # Add mandatory final trip to incinerator if needed
                    if has_collected_any_bins_this_truck_temp and current_load_temp > 0:
                        dist_to_inc_temp = calculate_distance(last_location_temp, incinerator_location)
                        current_truck_distance_temp += dist_to_inc_temp
                        calculated_incinerator_trips += 1 # Count this final trip
                        dist_inc_to_depot_temp = calculate_distance(incinerator_location, end_depot_location)
                        current_truck_distance_temp += dist_inc_to_depot_temp
                    else: # Ensure truck returns to depot if it didn't go to incinerator
                        if last_location_temp != end_depot_location:
                            dist_to_final_depot_temp = calculate_distance(last_location_temp, end_depot_location)
                            current_truck_distance_temp += dist_to_final_depot_temp

                    calculated_total_distance += current_truck_distance_temp

            accuracy = (theoretical_min_cost_benchmark / overall_best_fitness) * 100 if overall_best_fitness > 0 else 0

            # Convert tuple locations to lists for JSON serialization
            route_for_json = []
            if best_route_overall:
                for truck_route in best_route_overall:
                    single_truck_route_json = []
                    for stop_id in truck_route:
                        coords = get_location_coords(stop_id, bins_data_global, start_depot_location, incinerator_location, depot_id, incinerator_id)
                        single_truck_route_json.append({'id': stop_id, 'coords': list(coords) if coords else None})
                    route_for_json.append(single_truck_route_json)


            generation_results[str(generation + 1)] = { # Store as string key to match request
                'best_route': route_for_json,
                'fitness': float(overall_best_fitness),
                'total_distance': float(calculated_total_distance),
                'incinerator_trips': calculated_incinerator_trips,
                'trucks_used': calculated_trucks_used,
                'accuracy': float(accuracy)
            }
            print(f"Successfully saved data for generation {generation + 1}.")
            print(f"Saved best_route for gen {generation + 1}: {generation_results[str(generation + 1)]['best_route']}")
            print(f"Saved fitness for gen {generation + 1}: {generation_results[str(generation + 1)]['fitness']}")


        parents = select_mating_pool_vrp(new_population, fitness_scores, num_parents_mating)
        elite_individuals = []
        sorted_indices_for_elite = numpy.argsort(fitness_scores)
        for i in range(min(num_elite, sol_per_pop)):
            elite_individuals.append(new_population[sorted_indices_for_elite[i]])

        num_offspring = sol_per_pop - len(elite_individuals)
        if num_offspring < 0: num_offspring = 0

        offspring_crossover = crossover_vrp(parents, (num_offspring,), bin_ids_list, depot_id, incinerator_id)
        if len(offspring_crossover) < num_offspring:
            needed_more = num_offspring - len(offspring_crossover)
            for _ in range(needed_more):
                if parents:
                     offspring_crossover.append(list(random.choice(parents)))
                else:
                     offspring_crossover.append(create_initial_vrp_population(1,bin_ids_list,current_num_trucks,depot_id,incinerator_id, truck_capacity, bins_data_global)[0])


        offspring_mutation = mutation_vrp(offspring_crossover, mutation_rate, bin_ids_list, depot_id, incinerator_id)

        new_population = []
        new_population.extend(elite_individuals)
        new_population.extend(offspring_mutation)

        if len(new_population) > sol_per_pop:
            new_population = new_population[:sol_per_pop]
        elif len(new_population) < sol_per_pop:
            needed_to_fill = sol_per_pop - len(new_population)
            new_individuals = create_initial_vrp_population(needed_to_fill, bin_ids_list, current_num_trucks, depot_id, incinerator_id, truck_capacity, bins_data_global)
            new_population.extend(new_individuals)

    print("\n--- Final generation_results after GA run ---")
    if generation_results:
        for gen_key, data in generation_results.items():
            print(f"Generation {gen_key}: Fitness={data['fitness']:.2f}, Trucks={data['trucks_used']}, Route segments={len(data['best_route'][0]) if data['best_route'] else 'N/A'}")
    else:
        print("No generation results were stored.")
    print("---------------------------------------------")

    return jsonify({
        'message': 'GA simulation completed.',
        'theoretical_minimum_cost_benchmark': float(theoretical_min_cost_benchmark),
        'results_by_generation': generation_results # Sending the actual stored results back as well for inspection
    })

@app.route('/get_generations', methods=['GET'])
def get_generations():
    if not generation_results:
        return jsonify({"error": "GA has not been run yet. No generation data available."}), 400
    # Ensure keys are returned as strings if they were stored as strings
    return jsonify({"generations": sorted([int(gen) for gen in generation_results.keys()])})

@app.route('/get_route_data/<int:generation>', methods=['GET'])
def get_route_data(generation):
    # Retrieve using string key if stored as string
    if str(generation) not in generation_results:
        return jsonify({"error": f"Data for generation {generation} not found."}), 404
    return jsonify(generation_results[str(generation)])

# NEW ENDPOINT: To get the live fitness history
@app.route('/get_fitness_history', methods=['GET'])
def get_fitness_history():
    global fitness_history
    return jsonify({'fitness_values': fitness_history})

if __name__ == '__main__':
    app.run(debug=True, port=5000)