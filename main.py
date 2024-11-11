import os
import json
import webbrowser
import requests
import websocket
from websocket import WebSocketConnectionClosedException
from dotenv import load_dotenv
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Tuple, Set, List, Any
from itertools import islice
import pickle
import numpy as np
from heapq import heappush, heappop

load_dotenv()

@dataclass
class Node:
    coords: Tuple[int, int] = field(default_factory=tuple)  # (x, y)
    explored: bool = False
    distance: float = float('inf')
    surrounding_nodes: Set[Tuple[int, int]] = field(default_factory=set)  # {(x, y), (x, y)}

class GameClient:
    def __init__(self):
        self.PLAYER_TOKEN = os.getenv('PLAYER_TOKEN')
        self.LEVEL_ID = os.getenv('LEVEL_ID')
        self.FRONTEND_BASE = 'goldrush.monad.fi'
        self.BACKEND_BASE = 'goldrush.monad.fi/backend'
        self.optimal_path_file = f"{self.LEVEL_ID}_optimal_path.pkl"
        self.game_state = None
        self.ws = None
        self.entityId = None
        self.ready_to_start = threading.Event()
        self.shutdown_flag = threading.Event()
        self.initialized = False
        self.starting_coords = None
        
        self.found_nodes: Dict[Tuple[int, int], Node] = {} # coords: Node
        self.unvisited_coords = []
        self.all_unvisited_coords = []
        self.unknown_nodes: Dict[Tuple[int, int], Node] = {} # coords: Node
        self.all_nodes: Dict[Tuple[int, int], Node] = {}
        self.current_node = None
        self.closest_node = None
        self.goal_node = None
        self.optimal_path = None #self.load_optimal_path()
        self.running_optimal_path = False

        self.max_distance_to_goal = float('inf')
        self.current_distance_to_goal = float('inf')
        self.max_time = float('inf')
        self.current_time = float('inf')

        self.path_to_closest = [] # [(x,y), (x,y), ...]

        # only compute all unknown node distance every N times we open a unknown node
        # otherwise alternate between computing neighbouring nodes and updating close nodes
        # this helps with performance while having negligible impact on estimation accuracy
        # 0 = update all nodes all the time
        self.unknown_node_timer = None # set in the initialization
        # only adjust the copy value, easier to reset the timer this way
        self.time_copy = self.unknown_node_timer

        # ------------ parameters to adjust ------------ 
        
        # turns on some printing for logging purposes
        self.verbose = True

        # does a super comprehensive search for the most optimal path, SUPER compute intensive
        self.comprehensive_search = False

        # weight for preventing the player from hopping between 2 distant nodes
        # the weight is lowered as the player approaches the goal
        self.base_weight = 2 # 2
        # hard minimum for the distance weight
        self.min_distance_weight = 0.1 #0.05

        # weight based on time, the lower the remaining time
        # the more the players is discouraged from hopping from node to node
        self.max_time_weight = 1.2 # 0.2

        # scale for discounting long unknown paths, longer the path, the more we discount it
        # formula: unknown_path_len = unknown_path ** self.unknow_path_discount_factor
        # because 100 known + 50 unknow is better than 150 unknown
        self.unknow_path_discount_factor = 1.2 # 1.1

        # only check number of nodes with the lowest estimated distance from start to goal
        self.num_nodes_to_check = 10 # 20
        self.percent_nodes_to_update = 0.05 # 0.15


    def do_initialization(self, game_state):
        # set all the basic unchanging paramers when we see the game state for the first time

        self.starting_coords = tuple(game_state['start'].values())
        
        self.max_time = game_state["timer"]
        goal_coords = tuple(game_state["target"].values())
        self.goal_node = Node(coords=goal_coords, distance=0)

        rows = game_state["rows"]
        cols = game_state["columns"]
        self.unknown_nodes = self.generate_unknown_nodes(rows, cols)
        self.all_nodes = self.unknown_nodes.copy()

        if not self.comprehensive_search:
            self.unknown_node_timer = int(np.floor(rows * cols / 200))
        else:
            self.unknown_node_timer = 0
        self.time_copy = self.unknown_node_timer
            
        starting_distance = self.get_estimated_distance(self.starting_coords, self.goal_node.coords)
        self.max_distance_to_goal = starting_distance
        self.current_distance_to_goal = starting_distance

        if self.optimal_path is not None:
            self.path_to_closest = self.optimal_path

        self.initialized = True


    def load_optimal_path(self):
        # checks if optimal path has been found from a previous run
        # if optimal path found, we use that to navigate to the goal
        if os.path.exists(self.optimal_path_file):
            print("optimal path for level found")
            with open(self.optimal_path_file, 'rb') as file:
                return pickle.load(file)
        else:
            print("no optimal path found for level")
            return None
        
    def save_optimal_path(self):
        # calculates the optimal path to the goal to use in the next run
        if not os.path.exists(self.optimal_path_file) and self.optimal_path is None:
            print("Saving optimal path")
            estimate_len, optimal_path = self.a_star(self.found_nodes, self.starting_coords, self.goal_node.coords)
            print(f"estimated path length: {estimate_len}")
            with open(self.optimal_path_file, 'wb') as file:
                pickle.dump(optimal_path, file)

    def message(self, action: str, payload=None):
        return json.dumps([action, payload or {}])

    def get_walls(self, square: int) -> Dict[int, bool]:
        # gets the surrounding walls from current square number
        masks = [0b1000, 0b0100, 0b0010, 0b0001]
        return {
            0: (square & masks[0]) != 0,
            90: (square & masks[1]) != 0,
            180: (square & masks[2]) != 0,
            270: (square & masks[3]) != 0,
        }
    
    def generate_unknown_nodes(self, rows: int, columns: int):
        # adds x * y grid of unknown nodes with all linked to eachother
        unknown = dict()

        for x in range(columns):
            for y in range(rows):
                coords = (x,y)
                unknown[coords] = Node(coords=coords)

        valid_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        for coord, node in unknown.items():
            for diff in valid_neighbours:
                potential_neighbour = self.add_coords(coord, diff)
                if potential_neighbour in unknown:
                    node.surrounding_nodes.add(potential_neighbour)

        return unknown 
    
    def delete_unknow_coord(self, current_coords):
        # deletes a coord from self.unknown_nodes and all the connections to the node
        assert current_coords in self.unknown_nodes

        # delete neighbouring connections
        valid_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for diff in valid_neighbours:
            potential_neighbour = self.add_coords(current_coords, diff)
            if potential_neighbour in self.unknown_nodes:
                self.unknown_nodes[potential_neighbour].surrounding_nodes.remove(current_coords)

        # delete node itself
        del self.unknown_nodes[current_coords]

    def update_unknown_coord(self, current_node: Node):
        # changes a coord from self.all_nodes and all the connections to the node
        current_coords = current_node.coords
        neighbour_nodes = current_node.surrounding_nodes

        assert current_coords in self.all_nodes

        # delete neighbouring connections
        valid_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for diff in valid_neighbours:
            potential_neighbour = self.add_coords(current_coords, diff)
            if potential_neighbour in self.all_nodes and potential_neighbour not in neighbour_nodes:
                if current_coords in self.all_nodes[potential_neighbour].surrounding_nodes:
                    self.all_nodes[potential_neighbour].surrounding_nodes.remove(current_coords)

        # updates the node
        self.all_nodes[current_coords] = current_node
        
    
    def get_estimated_distance(self, point_a: Tuple[int, int], point_b: Tuple[int, int]):
        # gets the maximum difference in coordinates, better at estimating min distance when we can move diagonally
        x_1, y_1 = point_a
        x_2, y_2 = point_b
        return max((abs(x_1 - x_2), abs(y_1 - y_2)))
    
    def add_coords(self, point_a: Tuple[int, int], point_b: Tuple[int, int]):
        # adds 2 coords together
        x_1, y_1 = point_a
        x_2, y_2 = point_b
        return (x_1 + x_2, y_1 + y_2)
    
    def calculate_movement_cost(self, straigh_movement_coords: Tuple[int, int], neighbor_coords: Tuple[int, int]) -> int:
        # checks if the move has to change direction or not
        if straigh_movement_coords is None:
            return 1

        elif straigh_movement_coords == neighbor_coords:
             # we dont have to turn, so only a movement is required
            return 1
        
        else:
            # turning + movement
            return 2

        
    def get_best_coord_movement(self, previous_coords: Tuple[int, int], current_coords: Tuple[int, int]):
        # calculated the coordinates that we have to move to in order to not have a movement cost
        if previous_coords is None:
            return None
        previous_coords = np.array(previous_coords)
        current_coords = np.array(current_coords)
        diff = current_coords - previous_coords
        no_rotation_coord = current_coords + diff
        return tuple(no_rotation_coord)

    
    def a_star(self, node_dict: Dict[Tuple[int, int], Node], current_coords: Tuple[int, int], target_coords: Tuple[int, int]):
        # A star pathfinding algorithm that gets the path of coordinates from one point to the next

        if current_coords == target_coords:
            return 0, []
        
        open_set = []
        dist_to_target = self.get_estimated_distance(current_coords, target_coords)
        heappush(open_set, (dist_to_target, current_coords))
        came_from = dict()
     
        g_score = dict()
        f_score = dict()
        
        g_score[current_coords] = 0
        f_score[current_coords] = dist_to_target
        
        open_set_hash = set([current_coords])
        
        while open_set:
            _, current = heappop(open_set)

            if current == target_coords:
                # returns the total path len and the path from node to node
                return f_score.get(target_coords, float('inf')), self.reconstruct_path(came_from, current)
            
            previous_coords = came_from.get(current, None)

            open_set_hash.remove(current)
            
            straigh_movement_coords = self.get_best_coord_movement(previous_coords, current)

            all_neighbor_coords = node_dict[current].surrounding_nodes.copy()

            if previous_coords is not None:
                all_neighbor_coords.remove(previous_coords)
            
            for neighbor_coord in all_neighbor_coords:
                
                movement_cost = self.calculate_movement_cost(straigh_movement_coords, neighbor_coord)

                tentative_g_score = g_score.get(current, float('inf')) + movement_cost
                
                if tentative_g_score < g_score.get(neighbor_coord, float('inf')):
                    came_from[neighbor_coord] = current
                    g_score[neighbor_coord] = tentative_g_score
                    new_f_score = tentative_g_score + self.get_estimated_distance(neighbor_coord, target_coords)
                    f_score[neighbor_coord] = new_f_score
                    if neighbor_coord not in open_set_hash:
                        heappush(open_set, (new_f_score, neighbor_coord))
                        open_set_hash.add(neighbor_coord)
        
        # no path found
        return None, None
    
    def reconstruct_path(self, came_from, current):
        # support function for A star
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse() 
        return total_path[1:]
    
    def calculate_coord_diff(self, point_a, point_b):
        # returns the difference for coords
        x_1, y_1 = point_a
        x_2, y_2 = point_b
        return (x_1 - x_2, y_1 - y_2)
        

    def direction_to_coords(self, direction: int, coords: Tuple[int, int]) -> Tuple[int, int]:
        # converts the degree change to x and y change
        convert = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
        change = convert[direction]
        x, y = self.add_coords(coords, change)

        return (x, y)
    
    def get_surrounding_coords(self, current_coords: Tuple[int, int], square: int) -> Set[Tuple[int, int]]:
        # gets valid coordinates surrounding the player
        surrounding_coords = set()
        walls = self.get_walls(square)
        possible_directions = [rot for rot, wall in walls.items() if not wall]
        for dir in possible_directions:
            new_coords = self.direction_to_coords(dir, current_coords)
            surrounding_coords.add(new_coords)

        return surrounding_coords
    
    def get_diagonal_neighbours(self, current_coords: Tuple[int, int], surrounding_coords: set) -> Set[Tuple[int, int]]:
        # returns valid known diagonal neighbours that can be accessed by the node
        valid_diag_diff = ((-1, 1), (1, 1), (1, -1), (-1, -1))
        valid_diag_coords = set()
        for coord in surrounding_coords:
            if coord in self.found_nodes:
                # get neighbours neighbour coords
                neighbour_nodes = self.found_nodes[coord]
                new_neighbour_coords = neighbour_nodes.surrounding_nodes
                for neighbour_coords in new_neighbour_coords:
                    # if difference between neighbours neighbour coord is in valid_diag_neighbours
                    # that means that we can go diagonally there and we will add it to neighbours
                    diff = self.calculate_coord_diff(neighbour_coords, current_coords)
                    if diff in valid_diag_diff:
                        valid_diag_coords.add(neighbour_coords)

        
        return valid_diag_coords

    def update_surrounding_nodes(self, square: int, current_node: Node):
        # checks the nodes surrounding the player and updates their neighbours  
        
        current_coords = current_node.coords

        # get coords that are 1 away (1,0), (-1,0), etc that we can move to
        surrounding_coords = self.get_surrounding_coords(current_coords, square)

        # only gets known diagonal coords
        valid_diag_neighbours = self.get_diagonal_neighbours(current_coords, surrounding_coords)
        
        # add all valid neighbours together
        surrounding_coords.update(valid_diag_neighbours)

        current_node.surrounding_nodes.update(surrounding_coords)

        for new_coords in surrounding_coords:
            if new_coords in self.found_nodes:
                new_node = self.found_nodes[new_coords]
            else:
                new_node = self.add_new_node(new_coords, is_explored=False)

            new_node.surrounding_nodes.add(current_node.coords)

    def add_goal_node(self):
        # adds the goal node and updates the neighbours with connection to the goal
        self.add_new_node(self.goal_node.coords, is_explored=True)
        goal_coords = self.goal_node.coords
        valid_neighbours = ((-1, 0), (0, -1), (0, 1), (1, 0))

        goal_neighbours = set()
        for coord in valid_neighbours:
            potential_coord = self.add_coords(goal_coords, coord)
            if potential_coord in self.found_nodes and goal_coords in self.found_nodes[potential_coord].surrounding_nodes:
                goal_neighbours.add(potential_coord)

        valid_diag_neighbours = self.get_diagonal_neighbours(self.goal_node.coords, goal_neighbours)
        for coord in valid_diag_neighbours:
            self.found_nodes[coord].surrounding_nodes.add(goal_coords)


    def get_closest_unexplored(self) -> Dict[Tuple[int, int], Node]:
        # get self.num_nodes_to_check amount of coords: Node pairs that have least distance

        if not self.comprehensive_search:
            unexplored_nodes = {coord: self.found_nodes[coord] for coord in self.unvisited_coords}
            # Sort nodes by distance, taking only the top N with the shortest distance, this will save on computation
            # and it's very unlikely that the node wouldn't be in the top N shortest total distance from start to goal
            sorted_unexplored_nodes = dict(sorted(unexplored_nodes.items(), key=lambda item: item[1].distance))
            sorted_unexplored_nodes = dict(islice(sorted_unexplored_nodes.items(), self.num_nodes_to_check))
        else:
            sorted_unexplored_nodes = {coord: self.found_nodes[coord] for coord in self.all_unvisited_coords}

        return sorted_unexplored_nodes
    
    def get_unexplored_neighbours(self):
        # check surrounding nodes around the player and returns the unvisited ones
        neighbouring_coords = self.current_node.surrounding_nodes
        unexplored_neighbours = {coord: self.found_nodes[coord] for coord in neighbouring_coords if coord in self.unvisited_coords}
        return unexplored_neighbours

    
    def update_unvisited_node_distances(self):
        # update the top N umexplored nodes with the least distance, we dont care about visited node distances so they arent updated
        # this saves on some compute on big maps and we dont care about nodes that are far away being upto date 
        # since they woudnt be explored anyway
        # very compute expensive operation, but worth it in the end since we aren't limited by compute but
        # the amount of steps we take in the environment

        if self.time_copy == 0:
            # check all nodes when time hits 0, resets timer
            if not self.comprehensive_search:
                unexplored_nodes = {coord: self.found_nodes[coord] for coord in self.unvisited_coords}
            else:
                unexplored_nodes = {coord: self.found_nodes[coord] for coord in self.all_unvisited_coords}
            self.time_copy = self.unknown_node_timer
            
        elif self.time_copy % 2 == 1:
            # only check nearby unexplored nodes every other time we clear a node
            unexplored_nodes = self.get_unexplored_neighbours()
            self.time_copy -= 1

        elif self.time_copy % 2 == 0:
            # check nearby unexplored nodes and a set of self.unvisited_coords
            # by taking N amount of coords from the start and then 
            # extending with them we can cycle the list update coords 
            N_coords = int(np.ceil(len(self.unvisited_coords) * self.percent_nodes_to_update))
            N_coords = int(np.clip(N_coords, 4, self.num_nodes_to_check))
            coords_to_update, self.unvisited_coords = self.unvisited_coords[:N_coords], self.unvisited_coords[N_coords:]
            self.unvisited_coords.extend(coords_to_update)

            unexplored_nodes = {coord: self.found_nodes[coord] for coord in coords_to_update}
            unexplored_neighbours = self.get_unexplored_neighbours()

            unexplored_nodes.update(unexplored_neighbours)
            self.time_copy -= 1
            
        else:
            assert "timer error"
        
        for coords, current_node in unexplored_nodes.items():
            total_distance = self.total_distance_through_node(coords)
            # None = no valid path to the goal node
            # so we treat the node as being explored so that we dont unnecessarily check the node unless needed
            if total_distance is None:
                self.delete_unknow_coord(coords)
                current_node.explored = True
                self.unvisited_coords.remove(coords)
            else:
                current_node.distance = total_distance

    def update_closest_node(self):
        # updates the node that is estimated to be closest to the goal based on distance from player to node and node to goal
        min_score = float('inf')
        closest_node = self.found_nodes[self.starting_coords]
        
        unexplored_neighbours = self.get_unexplored_neighbours()
        if len(unexplored_neighbours) == 0:
            # if we dont have close neighbours, we check all the nodes instead of some of them
            # this helps with pathing and avoids unnecessary jumping
            nodes_to_check = {coord: self.found_nodes[coord] for coord in self.unvisited_coords}
        else:
            nodes_to_check = self.get_closest_unexplored()
            nodes_to_check.update(unexplored_neighbours)
        

        # if we see the goal, we want to make sure that we have to most optimal possible path
        goal_in_nodes = (self.goal_node.coords in nodes_to_check.keys())

        # ranges between 0 and self.max_time_weight depending on time left
        # the less time we have left, the less we should be wasting time on going between nodes
        weighted_time_remaining = self.max_time_weight * (1 - (self.current_time / self.max_time))

        for node_coords, node in nodes_to_check.items():

            # scale the player path down a little so it doesnt have as much impact, since we want the best path
            # but it needs to be found semi quickly without hopping between spaces
            player_path_len, _ = self.a_star(self.found_nodes, self.current_node.coords, node_coords)

            assert player_path_len is not None

            # we scale the path from player to the node throughout the game, so at the start, we care a lot about not wasting time
            # hopping between the nodes, but towards the end, we want it go closer and closer to 0
            # distance_weight values range from aound 1 to 0
            # the higher the base weight, the faster the initial search will be but at the risk of producing sub optimal final path
            if not goal_in_nodes:
                distance_weight = self.base_weight * (self.current_distance_to_goal / self.max_distance_to_goal)
                distance_weight = max([self.min_distance_weight, distance_weight])
                weighted_player_path = player_path_len * (distance_weight + weighted_time_remaining)
            else:
                weighted_player_path = 0

            score = node.distance + weighted_player_path
            
            if score < min_score:
                min_score = score
                closest_node = node
                

        self.closest_node = closest_node
    
    def total_distance_through_node(self, current_coords: Tuple[int, int]):
        # finds the total distance from start to the end when it goes through the node
        dist_from_start, _ = self.a_star(self.found_nodes, self.starting_coords, current_coords)

        if not self.comprehensive_search:
            dist_to_end, _ = self.a_star(self.unknown_nodes, current_coords, self.goal_node.coords)
        else:
            dist_to_end, _ = self.a_star(self.all_nodes, current_coords, self.goal_node.coords)

        # if the node is cut off from other other unknown nodes, we know that it cannot have a path to the goal
        # so we return none and handle the situation in the function that called this
        if dist_to_end is None:
            return None
            
        return dist_from_start + (dist_to_end ** self.unknow_path_discount_factor)


    def add_new_node(self, current_coords: Tuple[int, int], is_explored: bool) -> Node:
        # adds a new node class to coordinates and adds the node to the all nodes dictionary
        new_node = Node(coords=current_coords, explored=is_explored)
        self.found_nodes[current_coords] = new_node
        if not is_explored:
            self.unvisited_coords.append(current_coords)
            self.all_unvisited_coords.append(current_coords)
        elif current_coords in self.unvisited_coords:
            self.unvisited_coords.remove(current_coords)
            self.all_unvisited_coords.remove(current_coords)

        return new_node
    

    def generate_step(self, curr_rotation: int) -> Dict[str, Any]:
        # based on the x and y difference of current node and destination node, this generates the wanted rotation
        current_coords = self.current_node.coords
        assert len(self.path_to_closest) != 0
        next_coords = self.path_to_closest[0]
        x, y = self.calculate_coord_diff(next_coords, current_coords)
        # difference to angle conversion:
        #(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270, (1, -1): 45... etc.
        new_rotation = (90 + np.degrees(np.arctan2(y, x))) % 360

        # if we are already correctly rotated, then just move forward
        if new_rotation == curr_rotation:
            self.path_to_closest = self.path_to_closest[1:]
            return {
                'action': 'move',
            }

        else:
            return {
                'action': 'rotate',
                'rotation': new_rotation,
                }

    def generate_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        # update values based on game state
        player, square = game_state['player'], game_state['square']
        current_coords = tuple(player['position'].values())

        if not self.initialized:
            self.do_initialization(game_state)

        self.current_time = game_state["timer"]
        self.current_distance_to_goal = self.get_estimated_distance(current_coords, self.goal_node.coords)

        # check if we have entered a new or unexplored node
        if current_coords in self.found_nodes and self.found_nodes[current_coords].explored:
            self.current_node = self.found_nodes[current_coords]       
        else:
            # if the node is new, we delete it from unkonw,add it to all nodes
            # update surroundings and do new pathing calculations
            self.current_node = self.add_new_node(current_coords, is_explored=True)
            self.update_surrounding_nodes(square, self.current_node)
            if not self.comprehensive_search:
                self.delete_unknow_coord(current_coords)
            else:
                self.update_unknown_coord(self.current_node)
            # coordinate path to the node that is estimated to be closest to goal node
            if len(self.path_to_closest) == 0:
                self.update_unvisited_node_distances()
                self.update_closest_node()
                _, self.path_to_closest = self.a_star(self.found_nodes, self.current_node.coords, self.closest_node.coords)
   
        if not self.running_optimal_path and (self.path_to_closest[0] == self.goal_node.coords):
            self.add_goal_node()

            _, self.path_to_closest = self.a_star(self.found_nodes, self.starting_coords, self.goal_node.coords)
            next_move = {
                'action': 'reset',
            }
            self.running_optimal_path = True        
        elif self.current_node.coords != self.goal_node.coords:
            curr_rotation = player['rotation']
            # either turn to face correct block or move forward

            next_move = self.generate_step(curr_rotation)
        else:
            next_move = {
                '': '',
            }

        return next_move


    def create_game(self):
        # creates a game connection
        url = f'https://{self.BACKEND_BASE}/api/levels/{self.LEVEL_ID}'
        headers = {'Authorization': self.PLAYER_TOKEN}
        response = requests.post(url, headers=headers)

        if not response.ok:
            print(f"Couldn't create game: {response.status_code} - {response.text}")
            return None

        return response.json()

    def on_message(self, ws, message):
        # communicates with the server
        try:
            action, payload = json.loads(message)
            if action == 'game-instance':
                self.game_state = payload['gameState']  
                self.ready_to_start.set()
            else:
                if self.current_node.coords == self.goal_node.coords:
                    print("Game won")
                    self.save_optimal_path()
                    self.shutdown()
                else:
                    print(f"Unhandled action type: {action}")
        except Exception as e:
            print(f"Error processing message: {e}")

    def reconnect(self):
        # attempts to reconnect to the server
        print("Connection closed, attempting to reconnect...")
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
        
        # Re-establish the connection
        ws_url = f'wss://{self.BACKEND_BASE}/{self.PLAYER_TOKEN}/'
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_message=self.on_message,
                                         on_open=lambda ws: self.ws.send(self.message('sub-game', {'id': self.entityId})))
        threading.Thread(target=lambda: self.ws.run_forever()).start()


    def action_loop(self):
        self.ready_to_start.wait()
        last_message = time.time()
        last_command = None
        time_taken = 0
        while not self.shutdown_flag.is_set():
            start = time.perf_counter()
            if self.game_state:
                command = self.generate_action(json.loads(self.game_state))
                last_command = command
                self.send_command(command)
                self.game_state = None
                last_message = time.time()
            else:
                time_taken = time.time() - last_message
                if time_taken > 5 and last_command is not None:
                    if self.verbose:
                        print(f"last message was sent: {time_taken}")
                    self.send_command(last_command)
                    last_message = time.time()
                
                

            time_used = time.perf_counter() - start
            if time_used < 0.1:
                # small rate limit if needed
                time.sleep(0.1 - time_used)
    
    def send_command(self, command):
        for _ in range(5):
            try:
                self.ws.send(self.message('run-command', {'gameId': self.entityId, 'payload': command}))
                return
            except WebSocketConnectionClosedException:
                self.reconnect()
                time.sleep(0.1)

        print("failed to connect")
        self.shutdown()
                


    def shutdown(self):
        # signals all threads to stop
        self.shutdown_flag.set()  
        if self.ws:
            self.ws.close() 
            

    def start(self):
        game = self.create_game()
        if not game:
            return

        self.entityId = game['entityId'] 
        url = f'https://{self.FRONTEND_BASE}/?id={self.entityId}'
        print(f'Game at {url}')
        webbrowser.open(url)

        ws_url = f'wss://{self.BACKEND_BASE}/{self.PLAYER_TOKEN}/'
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_message=self.on_message,
                                         on_open=lambda ws: self.ws.send(self.message('sub-game', {'id': self.entityId})))

        threading.Thread(target=self.ws.run_forever).start()
        threading.Thread(target=self.action_loop).start()

        

if __name__ == "__main__":
    client = GameClient()
    client.start()