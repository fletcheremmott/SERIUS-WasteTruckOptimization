import numpy
import matplotlib.pyplot as plt
import math
import random
import heapq 

# -- Bins and Grid Requirements --
num_bins = 48
grid_range = 15 # 15x15 km grid
bin_volume_min = 360 # kg
bin_volume_max = 660 # kg
bin_service_time = 0.1 # hours, constant for all bins for simplicity

bins_data = {}
for i in range(num_bins):
    bin_id = f'{i+1}'
    loc_x = random.uniform(0, grid_range)
    loc_y = random.uniform(0, grid_range)
    volume = random.uniform(bin_volume_min, bin_volume_max)
    bins_data[bin_id] = {'loc': (loc_x, loc_y), 'volume': volume, 'service_time': bin_service_time}

totalTrash = sum(bin_info['volume'] for bin_info in bins_data.values())

# -- Definitions for the Trucks -- 
num_trucks_for_ga_start = 1
truck_capacity = 10000 # kg
truck_speed = 40  # units/hour (remains for time calculation, though time cost is zeroed)

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

# -- Utility Functions --
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

    closest_bin_to_inc_dist = calculate_closest_bin_distace_theoretical(bins_data, incinerator_loc, depot_loc, min_trips, cost_per_km)
    # Find the bin closest to the incinerator
    for bin_id, data in bins_data.items():
        dist = calculate_distance(data['loc'], incinerator_loc)
        if dist < closest_bin_to_inc_dist:
            closest_bin_to_inc_dist = dist

    # Total distance for one theoretical incinerator trip
    distance_per_trip = 2*closest_bin_to_inc_dist
    
    return min_trips * distance_per_trip * cost_per_km


# --- Core GA Functions for VRP ---

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
    
    # When num_trucks is fixed at 1, 'trucks_used' will always be 1 if the route is not empty
    # and contains at least one bin.
    trucks_used_in_chromosome = 0 

    all_bins_in_problem_local = set(bins_data.keys())
    visited_bins_in_chromosome = set()

    capacity_violation_penalty = 0
    unvisited_bin_penalty = 0
    invalid_bin_penalty_val = 0

    # Assuming 'chromosome' contains a list of routes, even if only one truck is used.
    # The current setup means `len(chromosome)` will be equal to `num_trucks_for_ga_start`.
    for truck_idx, truck_route in enumerate(chromosome):
        current_load = 0
        current_truck_distance = 0
        current_truck_time = 0
        last_location = start_depot_location
        has_collected_any_bins_this_truck = False

        # Determine if this truck is actually "used" for collecting bins.
        # This count is important for the `COST_PER_ADDITIONAL_TRUCK` logic.
        has_any_bins_in_route = any(stop_id in bins_data for stop_id in truck_route)
        if has_any_bins_in_route:
            trucks_used_in_chromosome += 1
        
        # If the route is empty or only depot-depot, it effectively doesn't perform service.
        # However, for a single truck GA, we *expect* it to service all bins.
        if not truck_route or (len(truck_route) == 2 and truck_route[0] == depot_id and truck_route[1] == depot_id):
            # If the only truck's route is empty, it means no bins are visited.
            # This will be caught by unvisited_bin_penalty.
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
    
    # Since we are strictly using 1 truck, the 'additional_truck_cost' will be 0.
    # The base cost for the first truck is assumed to be covered by other parts or is implicit.
    additional_truck_cost = max(0, trucks_used_in_chromosome) * cost_per_additional_truck

    # Combine all costs and penalties for fitness
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
        
        # This will create `num_trucks` sub-routes.
        # Since num_trucks is fixed at 1 for the GA, this loop will run once.
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
        
        # num_routes will always be 1 here because num_trucks_for_ga_start is 1
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
        mutated_chromosome = [list(route) for route in chromosome] # chromosome should only have 1 route (truck)
        
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
        # For 1 truck, this is effectively moving within the same route.
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
                
                target_truck_idx = random.randrange(len(mutated_chromosome)) # Still randomly pick, but it will be the same truck
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

        # Mutation 3: Add/Remove an incinerator trip (simple heuristic)
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

# -- Global variable for key press state --
space_pressed_event = False

def on_key_press(event):
    """Handles key press events for stepping through the plot."""
    global space_pressed_event
    if event.key == ' ':
        space_pressed_event = True
    elif event.key == 'escape':
        if event.canvas.figure.number == 2:
             plt.close(event.canvas.figure)
        space_pressed_event = True


# -- Plotting Function for Routes (Interactive) --
def plot_truck_routes_interactive(solution_chromosome, bins_data_dict, depot_loc, incinerator_loc_coords,
                                 depot_id_str, incinerator_id_str, fig_num=1):
    """Plots the truck routes interactively, stepping with space bar."""
    global space_pressed_event
    fig = plt.figure(fig_num, figsize=(12, 10))
    plt.clf()
    ax = plt.gca()

    # Set axis limits to ensure everything is visible
    all_x = [depot_loc[0], incinerator_loc_coords[0]] + [data['loc'][0] for data in bins_data_dict.values()]
    all_y = [depot_loc[1], incinerator_loc_coords[1]] + [data['loc'][1] for data in bins_data_dict.values()]
    
    min_x, max_x = min(all_x) - 1, max(all_x) + 1
    min_y, max_y = min(all_y) - 1, max(all_y) + 1
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.plot(depot_loc[0], depot_loc[1], 'ks', markersize=10, label='Depot')
    ax.text(depot_loc[0], depot_loc[1] + 0.2, 'Depot', ha='center', va='bottom', fontsize=9)
    ax.plot(incinerator_loc_coords[0], incinerator_loc_coords[1], 'm^', markersize=10, label='Incinerator')
    ax.text(incinerator_loc_coords[0], incinerator_loc_coords[1] + 0.2, 'INC', ha='center', va='bottom', fontsize=9)
    bin_label_added = False
    for bin_id, data in bins_data_dict.items():
        loc = data['loc']
        if not bin_label_added:
            ax.plot(loc[0], loc[1], 'bo', markersize=7, label='Bin')
            bin_label_added = True
        else:
            ax.plot(loc[0], loc[1], 'bo', markersize=7)
        ax.text(loc[0], loc[1] + 0.2, bin_id, ha='center', va='bottom', fontsize=8)
    route_colors = ['r', 'g', 'c', 'y', 'orange', 'purple', 'brown', 'pink', 'lime', 'darkblue']
    cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    print("\n--- Interactive Route Plotting ---")
    print("Focus the 'Truck Routes' plot window.")
    print("Press SPACE to draw the next route segment.")
    print("Press ESCAPE to close ONLY THIS route plot and continue/end script.")
    all_segments_plotted = False
    try:
        for truck_idx, route in enumerate(solution_chromosome):
            if not plt.fignum_exists(fig.number): break
            # Only plot if the route actually contains bins or incinerator trips
            has_any_bins_or_inc_in_route = any(stop_id in bins_data_dict or stop_id == incinerator_id for stop_id in route)
            if not has_any_bins_or_inc_in_route:
                continue

            color = route_colors[truck_idx % len(route_colors)]
            truck_label = f'Truck {truck_idx+1} Route'
            ax.plot([], [], linestyle='-', color=color, marker='.', label=truck_label, alpha=0.8, linewidth=1.5)
            for i in range(len(route) -1):
                if not plt.fignum_exists(fig.number): break
                start_stop_id = route[i]
                end_stop_id = route[i+1]
                start_coord = get_location_coords(start_stop_id, bins_data_dict, depot_loc,
                                                 incinerator_loc_coords, depot_id_str, incinerator_id_str)
                end_coord = get_location_coords(end_stop_id, bins_data_dict, depot_loc,
                                               incinerator_loc_coords, depot_id_str, incinerator_id_str)
                if start_coord and end_coord:
                    space_pressed_event = False
                    print(f"Truck {truck_idx+1}: Ready to draw segment {start_stop_id} -> {end_stop_id}. Press SPACE...")
                    while not space_pressed_event:
                        if not plt.fignum_exists(fig.number): break
                        plt.pause(0.05)
                    if not plt.fignum_exists(fig.number): break
                    ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]],
                            linestyle='-', color=color, marker='.', alpha=0.8, linewidth=1.5)
                    dx = (end_coord[0] - start_coord[0])
                    dy = (end_coord[1] - start_coord[1])
                    if dx != 0 or dy != 0:
                        ax.arrow(start_coord[0], start_coord[1],
                                  dx * 0.95, dy * 0.95,
                                  color=color, shape='full', lw=0,
                                  length_includes_head=True, head_width=0.15, alpha=0.6)
                    fig.canvas.draw_idle()
                else:
                    print(f"Warning: Could not get coordinates for segment {start_stop_id} -> {end_stop_id}")
            if not plt.fignum_exists(fig.number): break
        all_segments_plotted = True
    finally:
        fig.canvas.mpl_disconnect(cid)
    if plt.fignum_exists(fig.number):
        ax.set_xlabel("X-coordinate (km)")
        ax.set_ylabel("Y-coordinate (km)")
        ax.set_title("Truck Routes (Explicit Path) - Interactive Stepping Complete")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.canvas.draw_idle()
        if all_segments_plotted:
            print("All route segments drawn. This plot will remain open.")
            print("Close it manually to see the fitness plot (if generated) or end script.")
    else:
        print("Route plot was closed.")

# -- GA Parameters --
sol_per_pop = 100
num_parents_mating = 32
num_generations = 2000
mutation_rate = 0.25
num_elite = 5 # Number of top individuals to carry over

# Fitness weights now primarily for penalties, as monetary costs are explicit
fitness_weights_simplified = {
    'unvisited': 1000000,            # High penalty for not visiting all bins
    'capacity_violation': 50000,     # High penalty for exceeding truck capacity
    'invalid_bin': 200000            # High penalty for invalid bin references
}

# -- Main GA Loop --
print("Starting Genetic Algorithm for VRP...")
print(f"Number of bins: {len(bins_data)}")
print(f"Starting number of trucks in GA: {num_trucks_for_ga_start} (Fixed for verification)") # Clarify this is for GA init
print(f"Truck capacity: {truck_capacity} kg")

bin_ids_list = list(bins_data.keys())

# Define all_bins_in_problem for penalty checks in results
all_bins_in_problem = set(bins_data.keys())

# -- Calculate Minimum Theoretical Cost (Benchmark) --
mst_bins_depot_distance = calculate_mst_for_bins_and_depot(bins_data, start_depot_location)

min_incinerator_trips_theoretical = math.ceil(totalTrash / truck_capacity) if truck_capacity > 0 else 0

theoretical_incinerator_travel_cost = calculate_incinerator_round_trip_cost(
    bins_data, incinerator_location, start_depot_location, min_incinerator_trips_theoretical, COST_PER_KM
)

min_driving_cost_benchmark = (mst_bins_depot_distance * COST_PER_KM) + theoretical_incinerator_travel_cost
min_gate_fees_cost_benchmark = min_incinerator_trips_theoretical * GATE_FEE
# For the benchmark, if there's any trash, at least one truck is "used", incurring its base cost (COST_PER_ADDITIONAL_TRUCK)
min_truck_cost_benchmark = COST_PER_ADDITIONAL_TRUCK if totalTrash > 0 else 0 

minimum_theoretical_cost_benchmark = min_driving_cost_benchmark + min_gate_fees_cost_benchmark + min_truck_cost_benchmark

print(f"\n--- Theoretical Minimum Cost (Benchmark) ---")
print(f"MST (Bins & Depot) Distance: {mst_bins_depot_distance:.2f} km")
print(f"Theoretical Incinerator Round-Trip Driving Cost: S${theoretical_incinerator_travel_cost:.2f}")
print(f"Minimum Driving Cost Benchmark: S${min_driving_cost_benchmark:.2f}")
print(f"Minimum Incinerator Trips (theoretical): {min_incinerator_trips_theoretical}")
print(f"Minimum Gate Fees Cost: S${min_gate_fees_cost_benchmark:.2f}")
print(f"Minimum Truck Operating Cost (for at least one truck): S${min_truck_cost_benchmark:.2f}")
print(f"Combined Theoretical Minimum Cost: S${minimum_theoretical_cost_benchmark:.2f}")


print("\nCreating initial VRP population...")
current_num_trucks = num_trucks_for_ga_start # Fixed at 1 for now as per requirement
new_population = create_initial_vrp_population(sol_per_pop, bin_ids_list, current_num_trucks, depot_id, incinerator_id, truck_capacity, bins_data)

best_fitness_per_gen_for_plot = []
overall_best_fitness = float('inf')
best_route_overall = []

for generation in range(num_generations):
    print(f"\n--- Generation : {generation} (Trucks: {current_num_trucks}) ---")
    fitness_scores = calculate_fitness_for_population(
        new_population, bins_data, truck_capacity,
        incinerator_location, start_depot_location,
        end_depot_location, truck_speed,
        incinerator_unload_time, COST_PER_KM, GATE_FEE, COST_PER_ADDITIONAL_TRUCK, fitness_weights_simplified
    )
    
    current_best_fitness_in_gen = numpy.min(fitness_scores)
    
    if current_best_fitness_in_gen < overall_best_fitness:
        overall_best_fitness = current_best_fitness_in_gen
        best_idx_in_pop = numpy.argmin(fitness_scores)
        best_route_overall = new_population[best_idx_in_pop]
    
    best_fitness_per_gen_for_plot.append(overall_best_fitness)

    print(f"Best fitness in generation {generation}: S${current_best_fitness_in_gen:.2f}")
    print(f"Overall best fitness so far: S${overall_best_fitness:.2f}")
    
    # -- Adaptive heuristic to increase truck count has been REMOVED as per requirement --
    # This ensures the GA always operates with `num_trucks_for_ga_start` (i.e., 1 truck).

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
                 offspring_crossover.append(create_initial_vrp_population(1,bin_ids_list,current_num_trucks,depot_id,incinerator_id, truck_capacity, bins_data)[0])


    offspring_mutation = mutation_vrp(offspring_crossover, mutation_rate, bin_ids_list, depot_id, incinerator_id)
    
    new_population = []
    new_population.extend(elite_individuals)
    new_population.extend(offspring_mutation)

    if len(new_population) > sol_per_pop:
        new_population = new_population[:sol_per_pop]
    elif len(new_population) < sol_per_pop:
        needed_to_fill = sol_per_pop - len(new_population)
        new_individuals = create_initial_vrp_population(needed_to_fill, bin_ids_list, current_num_trucks, depot_id, incinerator_id, truck_capacity, bins_data)
        new_population.extend(new_individuals)


# --- Results ---
print("\n--- GA Finished ---")
if best_route_overall:
    final_best_fitness_details = calculate_fitness_vrp_simplified(
        best_route_overall, bins_data, truck_capacity,
        incinerator_location, start_depot_location,
        end_depot_location, truck_speed,
        incinerator_unload_time, COST_PER_KM, GATE_FEE, COST_PER_ADDITIONAL_TRUCK, fitness_weights_simplified
    )
    
    calculated_total_distance = 0
    calculated_incinerator_trips = 0
    calculated_trucks_used = 0 # Will be 1 if solution is valid and using the truck.

    for truck_route in best_route_overall: # This loop will only run once for 1 truck
        current_truck_distance_temp = 0
        current_load_temp = 0
        last_location_temp = start_depot_location
        has_collected_any_bins_this_truck_temp = False

        has_any_bins_in_route_temp = any(stop_id in bins_data for stop_id in truck_route)
        if has_any_bins_in_route_temp: # If the single truck route has bins, it's considered "used"
            calculated_trucks_used = 1 # Fixed to 1 for this context

        processed_route_temp = list(truck_route)
        if not processed_route_temp or processed_route_temp[0] != depot_id:
            processed_route_temp.insert(0, depot_id)

        for i, stop_id in enumerate(processed_route_temp):
            current_location_temp = get_location_coords(stop_id, bins_data, start_depot_location, incinerator_location, depot_id, incinerator_id)
            if current_location_temp:
                dist = calculate_distance(last_location_temp, current_location_temp)
                current_truck_distance_temp += dist
                last_location_temp = current_location_temp
            
            if stop_id == incinerator_id:
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

    closest_bin_theoretical = calculate_closest_bin_distace_theoretical(bins_data, incinerator_location, start_depot_location, min_incinerator_trips_theoretical, COST_PER_KM)

    mst_total_incinerator_trip_distance = 2 * closest_bin_theoretical * min_incinerator_trips_theoretical
    mst_total_distance = mst_bins_depot_distance + mst_total_incinerator_trip_distance
    
    print(f"\n--- Theoretical Minimum Cost (Benchmark) ---")
    print(f"MST (Bins & Depot) Distance: {mst_bins_depot_distance:.2f} km")
    print(f"Closest Bin Trip to Incinerator Distance: {closest_bin_theoretical:.2f} km")
    print(f"Total distance traveled: {mst_total_distance:.2f} km")
    print(f"Theoretical Incinerator Round-Trip Driving Cost: S${theoretical_incinerator_travel_cost:.2f}")
    print(f"Minimum Driving Cost Benchmark: S${min_driving_cost_benchmark:.2f}")
    print(f"Minimum Incinerator Trips (theoretical): {min_incinerator_trips_theoretical}")
    print(f"Minimum Gate Fees Cost: S${min_gate_fees_cost_benchmark:.2f}")
    print(f"Minimum Truck Operating Cost (for at least one truck): S${min_truck_cost_benchmark:.2f}")
    print(f"Combined Theoretical Minimum Cost: S${minimum_theoretical_cost_benchmark:.2f}")


    print("\nBest solution chromosome found: ")
    for i, route_part in enumerate(best_route_overall):
        print(f"  Truck {i+1}: {route_part}")
    
    print(f"\n--- Best Solution Metrics ---")
    print(f"Total distance traveled: {calculated_total_distance:.2f} km")
    print(f"Cost of driving: S${calculated_total_distance * COST_PER_KM:.2f}")
    print(f"Number of incinerator trips: {calculated_incinerator_trips}")
    print(f"Total gate fees: S${calculated_incinerator_trips * GATE_FEE:.2f}")
    print(f"Number of trucks used in best solution: {calculated_trucks_used}")
    # For a fixed 1 truck GA, this will always be S$0.00 (max(0, 1-1) * 200)
    print(f"Cost for additional trucks: S${max(0, calculated_trucks_used - 1) * COST_PER_ADDITIONAL_TRUCK:.2f}")
    print(f"Final Total Cost (Fitness): S${final_best_fitness_details:.2f}")

    capacity_penalty_check = 0
    unvisited_penalty_check = 0
    invalid_bin_penalty_check = 0
    
    visited_bins_in_best_sol = set()
    for truck_route in best_route_overall:
        current_load_for_penalty_check = 0
        for stop_id in truck_route:
            if stop_id == incinerator_id:
                current_load_for_penalty_check = 0
            elif stop_id == depot_id:
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
    
    print(f"Capacity Violation Penalty (should be 0 for valid solution): S${capacity_penalty_check:.2f}")
    print(f"Unvisited Bin Penalty (should be 0 for valid solution): S${unvisited_penalty_check:.2f}")
    print(f"Invalid Bin Penalty (should be 0 for valid solution): S${invalid_bin_penalty_check:.2f}")


else:
    print("No best route found (e.g., GA did not improve initial population, or no feasible solution).")

# --- Plotting Fitness ---
fig_fitness = plt.figure(1, figsize=(10, 6))
plt.plot(best_fitness_per_gen_for_plot)
plt.xlabel("Generation")
plt.ylabel("Overall Best Fitness (Total Cost in SGD)")
plt.title("VRP Cost Improvement Over Generations (Monotonic, 1 Truck)")
plt.grid(True)

# --- Plotting Truck Routes Interactively ---
if best_route_overall:
    plot_truck_routes_interactive(best_route_overall, bins_data, start_depot_location,
                                  incinerator_location, depot_id, incinerator_id, fig_num=2)
else:
    print("Skipping route plot as no best solution was determined.")

if plt.fignum_exists(fig_fitness.number):
    plt.show()
else:
    print("Fitness plot was not generated or was closed.")

#minimum_theoretical_cost_benchmark
#final_best_fitness_details
print("Theoretical Minimum should always be lower than Final Best Fitness, accuracy will always be under 100%")
print("Accuracy = (Minimum Theoretical Cost / Final Best Fitness) * 100 ")
accuracy = (minimum_theoretical_cost_benchmark / final_best_fitness_details) * 100
print("Accuracy: " + str(accuracy) + "%")


print("\nEnd of script.")