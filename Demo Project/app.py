from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import random
import math
import numpy as np # Use np for numpy operations
import heapq

# Add the directory containing WorkingAlgorithm.py to the Python path
# This assumes app.py is in the same directory as WorkingAlgorithm.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import functions from your WorkingAlgorithm.py
# We'll need to copy the relevant functions here or ensure they are properly imported.
# For simplicity and to avoid circular imports/issues with global variables in original script,
# I will copy the necessary functions and parameters directly into this Flask app.

# --- VRP Instance Data (Parameters - mirroring frontend defaults) ---
# These will be overridden by frontend input, but good to have defaults
DEFAULT_NUM_BINS = 48
DEFAULT_GRID_RANGE = 15 # 15x15 km grid
DEFAULT_BIN_VOLUME_MIN = 360 # kg
DEFAULT_BIN_VOLUME_MAX = 660 # kg
DEFAULT_BIN_SERVICE_TIME = 0.1 # hours

DEFAULT_TRUCK_CAPACITY = 10000 # kg
DEFAULT_TRUCK_SPEED = 40  # units/hour

DEFAULT_START_DEPOT_LOCATION = {'x': 0, 'y': 0}
DEFAULT_END_DEPOT_LOCATION = {'x': 0, 'y': 0}

DEFAULT_INCINERATOR_LOCATION = {'x': 30, 'y': 30}
DEFAULT_INCINERATOR_UNLOAD_TIME = 0.5  # hours
DEFAULT_INCINERATOR_ID = 'INC'
DEFAULT_DEPOT_ID = 'DEPOT'

# --- Cost Parameters (prices in SGD) ---
COST_PER_KM = 3.2382
GATE_FEE = 80
COST_PER_ADDITIONAL_TRUCK = 200

# --- Utility Functions (from WorkingAlgorithm.py) ---
def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two locations."""
    # Ensure loc1 and loc2 are dicts with 'x' and 'y'
    return np.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2)

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

# --- MST and Theoretical Cost Functions (from WorkingAlgorithm.py) ---
def calculate_mst_for_bins_and_depot(bins_data, depot_loc):
    nodes = {'depot': depot_loc}
    for bin_id, data in bins_data.items():
        nodes[bin_id] = data['loc']

    if not nodes:
        return 0

    min_cost = 0
    start_node_id = list(nodes.keys())[0]
    
    min_heap = [(0, start_node_id)]
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

def calculate_closest_bin_distace_theoretical(bins_data, incinerator_loc):
    theoretical_closest = float('inf')
    if not bins_data:
        return 0 # No bins, no closest distance

    for bin_id, data in bins_data.items():
        dist = calculate_distance(data['loc'], incinerator_loc)
        if dist < theoretical_closest:
            theoretical_closest = dist
        
    return theoretical_closest

def calculate_incinerator_round_trip_cost(bins_data, incinerator_loc, depot_loc, min_trips, cost_per_km):
    if min_trips == 0 or not bins_data:
        return 0

    inc_to_depot = calculate_distance(incinerator_loc, depot_loc)
    closest_bin_to_inc_dist = calculate_closest_bin_distace_theoretical(bins_data, incinerator_loc)
    
    # Total distance for one theoretical incinerator trip
    # (closest bin -> incinerator) + (incinerator -> depot)
    # The original Python code had 2*closest_bin_to_inc_dist for intermediate trips,
    # and then closest_bin_to_inc_dist + inc_to_depot for the final trip.
    # Let's align this with the benchmark logic from the Python script's theoretical cost.
    
    # The theoretical cost for incinerator trips:
    # Each trip involves going to INC and returning to DEPOT.
    # The 'closest_bin_to_inc_dist' is used as a proxy for the 'last mile to inc' for each trip.
    # So, for each trip, it's (closest_bin_to_inc_dist + dist_inc_to_depot)
    
    distance_per_trip = closest_bin_to_inc_dist + inc_to_depot
    
    return min_trips * distance_per_trip * cost_per_km


# --- Core GA Functions (from WorkingAlgorithm.py) ---

def calculate_fitness_vrp_simplified(chromosome, bins_data, truck_capacity, incinerator_location_coords,
                                     start_depot_location, end_depot_location, truck_speed,
                                     incinerator_unload_time, cost_per_km, gate_fee, cost_per_additional_truck, weights):
    total_distance = 0
    total_time = 0
    total_incinerator_trips = 0
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
            trucks_used_in_chromosome += 1
        
        if not truck_route or (len(truck_route) == 2 and truck_route[0] == DEFAULT_DEPOT_ID and truck_route[1] == DEFAULT_DEPOT_ID):
            continue

        processed_route = list(truck_route)
        if not processed_route or processed_route[0] != DEFAULT_DEPOT_ID:
            processed_route.insert(0, DEFAULT_DEPOT_ID)

        for i, stop_id in enumerate(processed_route):
            current_location = None

            if stop_id == DEFAULT_DEPOT_ID:
                if i == 0:
                    current_location = start_depot_location
                else:
                    current_location = end_depot_location
            elif stop_id == DEFAULT_INCINERATOR_ID:
                current_location = incinerator_location_coords
                total_incinerator_trips += 1
                current_truck_time += incinerator_unload_time
                current_load = 0
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

        if has_collected_any_bins_this_truck and current_load > 0:
            dist_to_inc = calculate_distance(last_location, incinerator_location_coords)
            current_truck_distance += dist_to_inc
            current_truck_time += (dist_to_inc / truck_speed if truck_speed > 0 else float('inf'))
            current_truck_time += incinerator_unload_time
            total_incinerator_trips += 1
            dist_inc_to_depot = calculate_distance(incinerator_location_coords, end_depot_location)
            current_truck_distance += dist_inc_to_depot
            current_truck_time += (dist_inc_to_depot / truck_speed if truck_speed > 0 else float('inf'))
        else:
            if last_location != end_depot_location:
                dist_to_final_depot = calculate_distance(last_location, end_depot_location)
                current_truck_distance += dist_to_final_depot
                current_truck_time += (dist_to_final_depot / truck_speed if truck_speed > 0 else float('inf'))

        total_distance += current_truck_distance
        total_time += current_truck_time

    unvisited_bins = all_bins_in_problem_local - visited_bins_in_chromosome
    if unvisited_bins:
        unvisited_bin_penalty = len(unvisited_bins) * weights.get('unvisited', 1000000)

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
    return np.array(fitness_scores)


def select_mating_pool_vrp(population, fitness, num_parents):
    parents = []
    sorted_indices = np.argsort(fitness)
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
                    continue
                cleaned_route.append(route[j])
            child.append(cleaned_route)
            
        offspring.append(child)
    return offspring


def mutation_vrp(offspring_crossover, mutation_rate, bins_data_keys, depot_id, incinerator_id):
    mutated_offspring = []
    all_bins_set = set(bins_data_keys)
    for chromosome in offspring_crossover:
        mutated_chromosome = [list(route) for route in chromosome]
        
        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            truck_idx_to_mutate = random.randrange(len(mutated_chromosome))
            route_to_mutate = mutated_chromosome[truck_idx_to_mutate]
            
            bin_indices_in_route = [i for i, stop in enumerate(route_to_mutate)
                                    if stop != depot_id and stop != incinerator_id]
            if len(bin_indices_in_route) >= 2:
                idx1_map, idx2_map = random.sample(bin_indices_in_route, 2)
                route_to_mutate[idx1_map], route_to_mutate[idx2_map] = route_to_mutate[idx2_map], route_to_mutate[idx1_map]

        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            
            candidate_bins_to_move = []
            for r_idx, route in enumerate(mutated_chromosome):
                for s_idx, stop in enumerate(route):
                    if stop in all_bins_set:
                        candidate_bins_to_move.append((r_idx, s_idx, stop))
            
            if candidate_bins_to_move:
                source_route_idx, source_stop_idx, bin_to_move = random.choice(candidate_bins_to_move)
                
                mutated_chromosome[source_route_idx].pop(source_stop_idx)
                
                target_truck_idx = random.randrange(len(mutated_chromosome))
                target_route = mutated_chromosome[target_truck_idx]
                
                insert_pos = random.randrange(1, max(2, len(target_route) - 1))
                target_route.insert(insert_pos, bin_to_move)
                
                if not target_route or target_route[0] != depot_id:
                    target_route.insert(0, depot_id)
                if target_route[-1] != depot_id:
                    target_route.append(depot_id)
                
                cleaned_target_route = [target_route[0]]
                for j in range(1, len(target_route)):
                    if (target_route[j] == depot_id and cleaned_target_route[-1] == depot_id) or \
                       (target_route[j] == incinerator_id and cleaned_target_route[-1] == incinerator_id):
                        continue
                    cleaned_target_route.append(target_route[j])
                mutated_chromosome[target_truck_idx] = cleaned_target_route

        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            truck_idx = random.randrange(len(mutated_chromosome))
            route = mutated_chromosome[truck_idx]
            
            if random.random() < 0.5 and len(route) > 2:
                insert_pos = random.randrange(1, len(route) - 1)
                if route[insert_pos] != incinerator_id and route[insert_pos-1] != incinerator_id:
                    route.insert(insert_pos, incinerator_id)
            else:
                inc_indices = [i for i, stop in enumerate(route) if stop == incinerator_id and i != 0 and i != len(route)-1]
                if inc_indices:
                    route.pop(random.choice(inc_indices))

        mutated_offspring.append(mutated_chromosome)
    return mutated_offspring


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/run_vrp_ga', methods=['POST'])
def run_vrp_ga():
    data = request.json
    
    # Extract parameters from the request
    num_bins = data.get('num_bins', DEFAULT_NUM_BINS)
    grid_range = data.get('grid_range', DEFAULT_GRID_RANGE)
    bin_volume_min = data.get('bin_volume_min', DEFAULT_BIN_VOLUME_MIN)
    bin_volume_max = data.get('bin_volume_max', DEFAULT_BIN_VOLUME_MAX)
    truck_capacity = data.get('truck_capacity', DEFAULT_TRUCK_CAPACITY)
    num_generations = data.get('num_generations', 100) # Default for GA
    sol_per_pop = data.get('sol_per_pop', 100) # Default for GA
    mutation_rate = data.get('mutation_rate', 0.25)
    num_elite = data.get('num_elite', 5)

    # Re-generate bins data based on received parameters
    bins_data = {}
    totalTrash = 0
    for i in range(num_bins):
        bin_id = f'bin{i+1}'
        loc_x = random.uniform(0, grid_range)
        loc_y = random.uniform(0, grid_range)
        volume = random.uniform(bin_volume_min, bin_volume_max)
        bins_data[bin_id] = {'loc': {'x': loc_x, 'y': loc_y}, 'volume': volume, 'service_time': DEFAULT_BIN_SERVICE_TIME}
        totalTrash += volume

    # Define fixed locations and IDs
    start_depot_location = {'x': 0, 'y': 0}
    end_depot_location = {'x': 0, 'y': 0}
    incinerator_location = {'x': 30, 'y': 30}
    
    # GA parameters
    num_trucks_for_ga = 1 # Fixed to 1 as per latest requirement
    num_parents_mating = int(sol_per_pop * 0.32) # Roughly 32% of population

    # Fitness weights (penalties)
    fitness_weights_simplified = {
        'unvisited': 1000000,
        'capacity_violation': 50000,
        'invalid_bin': 200000
    }

    # --- Run the GA ---
    bin_ids_list = list(bins_data.keys())
    all_bins_in_problem = set(bins_data.keys())

    new_population = create_initial_vrp_population(sol_per_pop, bin_ids_list, num_trucks_for_ga, DEFAULT_DEPOT_ID, DEFAULT_INCINERATOR_ID, truck_capacity, bins_data)

    overall_best_fitness = float('inf')
    best_route_overall = []

    for generation in range(num_generations):
        fitness_scores = calculate_fitness_for_population(
            new_population, bins_data, truck_capacity,
            incinerator_location, start_depot_location,
            end_depot_location, DEFAULT_TRUCK_SPEED,
            DEFAULT_INCINERATOR_UNLOAD_TIME, COST_PER_KM, GATE_FEE, COST_PER_ADDITIONAL_TRUCK, fitness_weights_simplified
        )
        
        current_best_fitness_in_gen = np.min(fitness_scores)
        
        if current_best_fitness_in_gen < overall_best_fitness:
            overall_best_fitness = current_best_fitness_in_gen
            best_idx_in_pop = np.argmin(fitness_scores)
            best_route_overall = new_population[best_idx_in_pop]
        
        parents = select_mating_pool_vrp(new_population, fitness_scores, num_parents_mating)

        elite_individuals = []
        sorted_indices_for_elite = np.argsort(fitness_scores)
        for i in range(min(num_elite, sol_per_pop)):
            elite_individuals.append(new_population[sorted_indices_for_elite[i]])

        num_offspring = sol_per_pop - len(elite_individuals)
        if num_offspring < 0: num_offspring = 0

        offspring_crossover = crossover_vrp(parents, (num_offspring,), bin_ids_list, DEFAULT_DEPOT_ID, DEFAULT_INCINERATOR_ID)

        if len(offspring_crossover) < num_offspring:
            needed_more = num_offspring - len(offspring_crossover)
            for _ in range(needed_more):
                if parents:
                    offspring_crossover.append(list(random.choice(parents)))
                else:
                    offspring_crossover.append(create_initial_vrp_population(1,bin_ids_list,num_trucks_for_ga,DEFAULT_DEPOT_ID,DEFAULT_INCINERATOR_ID, truck_capacity, bins_data)[0])

        offspring_mutation = mutation_vrp(offspring_crossover, mutation_rate, bin_ids_list, DEFAULT_DEPOT_ID, DEFAULT_INCINERATOR_ID)
        
        new_population = []
        new_population.extend(elite_individuals)
        new_population.extend(offspring_mutation)

        if len(new_population) > sol_per_pop:
            new_population = new_population[:sol_per_pop]
        elif len(new_population) < sol_per_pop:
            needed_to_fill = sol_per_pop - len(new_population)
            new_individuals = create_initial_vrp_population(needed_to_fill, bin_ids_list, num_trucks_for_ga, DEFAULT_DEPOT_ID, DEFAULT_INCINERATOR_ID, truck_capacity, bins_data)
            new_population.extend(new_individuals)

    # --- Prepare results for frontend ---
    final_best_fitness_details = overall_best_fitness
    
    calculated_total_distance = 0
    calculated_incinerator_trips = 0
    calculated_trucks_used = 0

    for truck_route in best_route_overall:
        current_truck_distance_temp = 0
        current_load_temp = 0
        last_location_temp = start_depot_location
        has_collected_any_bins_this_truck_temp = False

        has_any_bins_in_route_temp = any(stop_id in bins_data for stop_id in truck_route)
        if has_any_bins_in_route_temp:
            calculated_trucks_used = 1 # Fixed to 1 for this context

        processed_route_temp = list(truck_route)
        if not processed_route_temp or processed_route_temp[0] != DEFAULT_DEPOT_ID:
            processed_route_temp.insert(0, DEFAULT_DEPOT_ID)

        for i, stop_id in enumerate(processed_route_temp):
            current_location_temp = get_location_coords(stop_id, bins_data, start_depot_location, incinerator_location, DEFAULT_DEPOT_ID, DEFAULT_INCINERATOR_ID)
            if current_location_temp:
                dist = calculate_distance(last_location_temp, current_location_temp)
                current_truck_distance_temp += dist
                last_location_temp = current_location_temp
            
            if stop_id == DEFAULT_INCINERATOR_ID:
                calculated_incinerator_trips += 1
                current_load_temp = 0
            elif stop_id in bins_data:
                current_load_temp += bins_data[stop_id]['volume']
                has_collected_any_bins_this_truck_temp = True
        
        if has_collected_any_bins_this_truck_temp and current_load_temp > 0:
            dist_to_inc_temp = calculate_distance(last_location_temp, incinerator_location)
            current_truck_distance_temp += dist_to_inc_temp
            calculated_incinerator_trips += 1
            dist_inc_to_depot_temp = calculate_distance(incinerator_location, end_depot_location)
            current_truck_distance_temp += dist_inc_to_depot_temp
        else:
            if last_location_temp != end_depot_location:
                dist_to_final_depot_temp = calculate_distance(last_location_temp, end_depot_location)
                current_truck_distance_temp += dist_to_final_depot_temp

        calculated_total_distance += current_truck_distance_temp

    # Calculate penalties for the best route to display
    capacity_penalty_check = 0
    unvisited_penalty_check = 0
    invalid_bin_penalty_check = 0
    
    visited_bins_in_best_sol = set()
    for truck_route in best_route_overall:
        current_load_for_penalty_check = 0
        for stop_id in truck_route:
            if stop_id == DEFAULT_INCINERATOR_ID:
                current_load_for_penalty_check = 0
            elif stop_id == DEFAULT_DEPOT_ID:
                pass
            elif stop_id in bins_data:
                bin_info = bins_data[stop_id]
                if current_load_for_penalty_check + bin_info['volume'] > truck_capacity:
                    capacity_penalty_check += (current_load_for_penalty_check + bin_info['volume'] - truck_capacity) * fitness_weights_simplified['capacity_violation']
                current_load_for_penalty_check += bin_info['volume']
                visited_bins_in_best_sol.add(stop_id)
            else:
                invalid_bin_penalty_check += fitness_weights_simplified['invalid_bin']
    
    unvisited_bins_check = all_bins_in_problem - visited_bins_in_best_sol
    unvisited_penalty_check = len(unvisited_bins_check) * fitness_weights_simplified['unvisited']

    # Convert bins_data and best_route_overall to a JSON-serializable format
    # Locations are already dicts, so they should be fine.
    serializable_bins_data = {
        bin_id: {
            'loc': {'x': data['loc']['x'], 'y': data['loc']['y']},
            'volume': data['volume'],
            'service_time': data['service_time']
        }
        for bin_id, data in bins_data.items()
    }

    response_data = {
        'binsData': serializable_bins_data,
        'depotLocation': start_depot_location,
        'incineratorLocation': incinerator_location,
        'routeResult': {
            'route': best_route_overall[0], # Assuming single truck, take the first route
            'totalDistance': calculated_total_distance,
            'totalIncineratorTrips': calculated_incinerator_trips,
            'totalCost': final_best_fitness_details,
            'trucksUsed': calculated_trucks_used,
            'capacityPenalty': capacity_penalty_check,
            'unvisitedPenalty': unvisited_penalty_check,
            'invalidBinPenalty': invalid_bin_penalty_check
        },
        'totalTrash': totalTrash,
        'truckCapacity': truck_capacity,
        'gaParameters': {
            'num_generations': num_generations,
            'sol_per_pop': sol_per_pop,
            'mutation_rate': mutation_rate,
            'num_elite': num_elite
        }
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run on port 5000
