import os
import json
import webbrowser
import requests
import websocket
from dotenv import load_dotenv
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import distance
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
        self.unknown_nodes: Dict[Tuple[int, int], Node] = {} # coords: Node
        self.current_node = None
        self.closest_node = None
        self.goal_node = None
        self.optimal_path = self.load_optimal_path()

        self.max_distance_to_goal = float('inf')
        self.current_distance_to_goal = float('inf')
        self.max_time = None
        self.current_time = 10000

        self.path_to_closest = [] # [(x,y), (x,y), ...]

        # ------------ parameters to adjust ------------ 

        # weight for preventing the player from hopping between 2 distant nodes
        # the weight is lowered as the player approaches the goal
        self.base_weight = 0.75
        # hard minimum for the distance weight
        self.min_distance_weight = 0.05

        # weight based on time, the lower the remaining time
        # the more the players is discouraged from hopping from node to node
        self.max_time_weight = 0.15

        # scale for discounting long unknown paths, longer the path, the more we discount it
        # formula: unknown_path_len = unknown_path ** self.unknow_path_discount_factor
        # because 100 known + 50 unknow is better than 150 unknown
        self.unknow_path_discount_factor = 1.05

        # only check number of nodes with the lowest estimated distance from start to goal
        self.num_nodes_to_check = 15

        # only compute all unknown node distance every N times we open a unknown node
        # otherwise only compute close unknown nodes, saves on compute
        self.unknown_node_timer = 10


    def do_initialization(self, game_state):
        # set all the basic unchanging paramers when we see the game state for the first time

        self.starting_coords = tuple(game_state['start'].values())
        
        self.max_time = game_state["timer"]
        goal_coords = tuple(game_state["target"].values())
        self.goal_node = Node(coords=goal_coords, distance=0)

        self.unknown_nodes = self.generate_unknown_nodes(game_state["rows"], game_state["columns"])

        self.max_distance_to_goal = self.get_estimated_distance(self.starting_coords, self.goal_node.coords)

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
                potential_neighbour = tuple(np.array(coord) + diff)
                if potential_neighbour in unknown:
                    node.surrounding_nodes.add(potential_neighbour)

        return unknown 
    
    def delete_unknow_coord(self, current_coords):
        assert current_coords in self.unknown_nodes

        # delete neighbouring connections
        valid_neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for diff in valid_neighbours:
            potential_neighbour = tuple(np.array(current_coords) + diff)
            if potential_neighbour in self.unknown_nodes:
                self.unknown_nodes[potential_neighbour].surrounding_nodes.remove(current_coords)

        # delete node itself
        del self.unknown_nodes[current_coords]
        
    
    def get_estimated_distance(self, point_a: Tuple[int, int], point_b: Tuple[int, int]):
        # more accurate than euclidean distance in a grid
        return np.linalg.norm(np.array(point_b) - np.array(point_a))
    
    def calculate_movement_cost(self, previous_coords: Tuple[int, int], current_coords: Tuple[int, int], next_coords: Tuple[int, int]) -> int:
        if previous_coords is None:
            return 1
        
        previous_diff = self.calculate_coord_diff(current_coords, previous_coords)

        current_diff = self.calculate_coord_diff(next_coords, current_coords)

        # we dont have to turn, so only a movement is required
        if current_diff == previous_diff:
            return 1
        else:
            # turning + movement
            return 2
        
    
    def a_star(self, node_dict, current_coords, target_coords):

        if current_coords == target_coords:
            return 0, []
        
        open_set = []
        heappush(open_set, (0 + self.get_estimated_distance(current_coords, target_coords), current_coords))
        came_from = {}
        
        g_score = {node: float('inf') for node in node_dict}
        f_score = g_score.copy()
        
        g_score[current_coords] = 0
        f_score[current_coords] = self.get_estimated_distance(current_coords, target_coords)
        
        open_set_hash = set([current_coords])
        
        while open_set:
            _, current = heappop(open_set)

            previous_coords = came_from.get(current, None)

            open_set_hash.remove(current)
            
            if current == target_coords:
                # returns the total path len and the path from node to node
                return f_score[target_coords], self.reconstruct_path(came_from, current)
            
            for neighbor_coords in node_dict[current].surrounding_nodes:

                movement_cost = self.calculate_movement_cost(previous_coords, current, neighbor_coords)

                tentative_g_score = g_score[current] + movement_cost
                
                if tentative_g_score < g_score.get(neighbor_coords, float('inf')):
                    came_from[neighbor_coords] = current
                    g_score[neighbor_coords] = tentative_g_score
                    f_score[neighbor_coords] = tentative_g_score + self.get_estimated_distance(neighbor_coords, target_coords)
                    if neighbor_coords not in open_set_hash:
                        heappush(open_set, (f_score[neighbor_coords], neighbor_coords))
                        open_set_hash.add(neighbor_coords)
        
        # no path found
        return None, None
    
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse() 
        return total_path[1:]
    
    def calculate_coord_diff(self, coords_1, coords_2):
        return tuple(np.array(coords_1) - np.array(coords_2))
        

    def direction_to_coords(self, direction: int, coords: Tuple[int, int]) -> Tuple[int, int]:
        # converts the degree change to x and y change
        convert = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
        change = convert[direction]
        x, y = np.array(coords) + change

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

        nodes_to_update = []
        for new_coords in surrounding_coords:
            if new_coords in self.found_nodes:
                new_node = self.found_nodes[new_coords]
            else:
                new_node = self.add_new_node(new_coords, is_explored=False)

            new_node.surrounding_nodes.add(current_node.coords)
            nodes_to_update.append(new_node)

    def get_closest_unexplored(self) -> Dict[Tuple[int, int], Node]:
        # get self.num_nodes_to_check amount of coords: Node pairs that have least distance

        unexplored_nodes = {coord: self.found_nodes[coord] for coord in self.unvisited_coords}
        # Sort nodes by distance, taking only the top N with the shortest distance, this will save on computation
        # and it's very unlikely that the node wouldn't be in the top N shortest total distance from start to goal
        sorted_unexplored_nodes = dict(sorted(unexplored_nodes.items(), key=lambda item: item[1].distance))
        sorted_unexplored_nodes = dict(islice(sorted_unexplored_nodes.items(), self.num_nodes_to_check))

        return sorted_unexplored_nodes
    
    
    def update_unvisited_node_distances(self):
        # update the top N umexplored nodes with the least distance, we dont care about visited node distances so they arent updated
        # this saves on some compute on big maps and we dont care about nodes that are far away being upto date 
        # since they woudnt be explored anyway
        # very compute expensive operation, but worth it in the end since we aren't limited by compute but
        # the amount of steps we take in the environment

        if self.unknown_node_timer == 0:
            # check all unexplored nodes, only happens once timer runs to 0
            unexplored_nodes = self.get_closest_unexplored()
            self.unknown_node_timer = 10
            print("check all")
        else:
            # only check near by unexplored nodes
            neighbouring_coords = self.current_node.surrounding_nodes
            unexplored_nodes = {coord: self.found_nodes[coord] for coord in neighbouring_coords if coord in self.unvisited_coords}
            self.unknown_node_timer -= 1
            print("check close")

        
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
            print()


    def update_closest_node(self):
        # updates the node that is estimated to be closest to the goal based on distance from player to node and node to goal
        min_score = float('inf')
        closest_node = self.found_nodes[self.starting_coords]

        sorted_unexplored_nodes = self.get_closest_unexplored()
        
        # ranges between 0 and self.max_time_weight depending on time left
        # the less time we have left, the less we should be wasting time on going between nodes
        weighted_time_remaining = self.max_time_weight * (1 - (self.current_time / self.max_time))

        for node_coords, node in sorted_unexplored_nodes.items():
            # scale the player path down a little so it doesnt have as much impact, since we want the best path
            # but it needs to be found semi quickly without hopping between spaces
            player_path_len, _ = self.a_star(self.found_nodes, self.current_node.coords, node_coords)

            # we scale the path from player to the node throughout the game, so at the start, we care a lot about not wasting time
            # hopping between the nodes, but towards the end, we want it go closer and closer to 0
            # distance_weight values range from aound 1 to 0
            # the higher the base weight, the faster the initial search will be but at the risk of producing sub optimal final path
            
            distance_weight = self.base_weight * (self.current_distance_to_goal / self.max_distance_to_goal)
            distance_weight = max([self.min_distance_weight, distance_weight])
            weighted_player_path = (player_path_len * distance_weight) + (player_path_len * weighted_time_remaining)
            score = node.distance + weighted_player_path

            if score < min_score:
                min_score = score
                closest_node = node

        
        self.closest_node = closest_node
    
    def total_distance_through_node(self, current_coords: Tuple[int, int]):
    
        dist_from_start, _ = self.a_star(self.found_nodes, self.starting_coords, current_coords)
        dist_to_end, _ = self.a_star(self.unknown_nodes, current_coords, self.goal_node.coords)

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
        elif current_coords in self.unvisited_coords:
            self.unvisited_coords.remove(current_coords)

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

        self.current_distance = self.get_estimated_distance(current_coords, self.goal_node.coords)

        # check if we have entered a new or unexplored node
        if current_coords in self.found_nodes and self.found_nodes[current_coords].explored:
            self.current_node = self.found_nodes[current_coords]       
        else:
            # if the node is new, we delete it from unkonw,add it to all nodes
            # update surroundings and do new pathing calculations
            self.delete_unknow_coord(current_coords)
            self.current_node = self.add_new_node(current_coords, is_explored=True)
            self.update_surrounding_nodes(square, self.current_node)
            self.update_unvisited_node_distances()
            self.update_closest_node()
            # coordinate path to the node that is estimated to be closest to goal node
            if len(self.path_to_closest) == 0:
                _, self.path_to_closest = self.a_star(self.found_nodes, self.current_node.coords, self.closest_node.coords)
   
        curr_rotation = player['rotation']

        # either turn to face correct block or move forward
        next_move = self.generate_step(curr_rotation)

        return next_move


    def create_game(self):
        url = f'https://{self.BACKEND_BASE}/api/levels/{self.LEVEL_ID}'
        headers = {'Authorization': self.PLAYER_TOKEN}
        response = requests.post(url, headers=headers)

        if not response.ok:
            print(f"Couldn't create game: {response.status_code} - {response.text}")
            return None

        return response.json()

    def on_message(self, ws, message):

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


    def action_loop(self):
        self.ready_to_start.wait()
        while not self.shutdown_flag.is_set():
            time.sleep(0.1)
            if self.game_state:
                command = self.generate_action(json.loads(self.game_state))  
                self.ws.send(self.message('run-command', {'gameId': self.entityId, 'payload': command}))
                self.game_state = None
    
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