import os
import json
import random
import webbrowser
import requests
import websocket
from dotenv import load_dotenv
import threading
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import distance

load_dotenv()

@dataclass
class Node():
    coords: tuple = field(default_factory=tuple) # (x, y)
    explored: bool = False
    distance: float = float('inf')  
    surrounding_nodes: set = field(default_factory=set) # {(x, y), (x, y)}

class GameClient:
    def __init__(self):
        self.PLAYER_TOKEN = os.getenv('PLAYER_TOKEN')
        self.LEVEL_ID = os.getenv('LEVEL_ID')
        self.FRONTEND_BASE = 'goldrush.monad.fi'
        self.BACKEND_BASE = 'goldrush.monad.fi/backend'
        self.game_state = None
        self.ws = None
        self.entityId = None
        self.ready_to_start = threading.Event()
        self.initialized = False
        
        self.all_nodes = {} # coords: Node
        self.current_node = None
        self.closest_node = None
        self.goal_node = None

        self.path_to_closest = []


    def message(self, action, payload=None):
        return json.dumps([action, payload or {}])

    def get_walls(self, square):
        masks = [0b1000, 0b0100, 0b0010, 0b0001]
        return {
            0: (square & masks[0]) != 0,
            90: (square & masks[1]) != 0,
            180: (square & masks[2]) != 0,
            270: (square & masks[3]) != 0,
        }
    
    def update_all_node_connections(self):
        for current_coords, current_node in self.all_nodes.items():
            x, y = current_coords 
            potential_surroundings = [
                (x + 1, y),  
                (x - 1, y),  
                (x, y + 1),  
                (x, y - 1),  
            ]
            for new_coord in potential_surroundings:
                if new_coord in self.all_nodes:
                    surrounding_node = self.all_nodes[new_coord]
                    current_node.surrounding_nodes.add(new_coord)
                    surrounding_node.surrounding_nodes.add(current_coords)
    
    def a_star(self, target_coords):
        open_set = set([self.current_node.coords])  # Start with the current node
        came_from = {}  # For path reconstruction

        g_score = {node: float('inf') for node in self.all_nodes}  # default score to infinity
        g_score[self.current_node.coords] = 0  # Cost from start to start is zero

        f_score = {node: float('inf') for node in self.all_nodes}  # default score to infinity
        f_score[self.current_node.coords] = distance.euclidean(self.current_node.coords, target_coords)  # Heuristic cost estimate to goal

        while open_set:
            # Node in open set with the lowest f_score
            current = min(open_set, key=lambda coord: f_score[coord])
            if current == target_coords:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor_coords in self.all_nodes[current].surrounding_nodes:
                tentative_g_score = g_score[current] + distance.euclidean(current, neighbor_coords)

                if tentative_g_score < g_score[neighbor_coords]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor_coords] = current
                    g_score[neighbor_coords] = tentative_g_score
                    f_score[neighbor_coords] = g_score[neighbor_coords] + distance.euclidean(neighbor_coords, target_coords)
                    if neighbor_coords not in open_set:
                        open_set.add(neighbor_coords)

        # empty path if no path found
        return []

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path[1:]


    def direction_to_coords(self, direction, coords):
        convert = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}
        change = convert[direction]
        x = coords[0] + change[0]
        y = coords[1] + change[1]

        return (x, y)
    
    def get_surrounding_coords(self, player, square):
        surrounding_coords = []
        walls = self.get_walls(square)
        position = tuple(player["position"].values())
        possible_directions = [rot for rot, wall in walls.items() if not wall]

        for dir in possible_directions:
            new_coords = self.direction_to_coords(dir, position)
            surrounding_coords.append(new_coords)

        return surrounding_coords

    def check_for_surrounding_nodes(self, player, square, current_node: Node):
        surrounding_coords = self.get_surrounding_coords(player, square)
        current_node.surrounding_nodes.update(surrounding_coords)

        surrounding_nodes = []
        for new_coords in surrounding_coords:

            if new_coords in self.all_nodes:
                new_node = self.all_nodes[new_coords]     
            else:
                new_node = self.add_new_node(new_coords, is_explored=False)

            new_node.surrounding_nodes.add(current_node.coords)
            surrounding_nodes.append(new_node)

        return surrounding_nodes
    
    
    '''def update_closest_node(self):
        unexplored_nodes = {node_id: node for node_id, node in self.all_nodes.items() if not node.explored}
        
        if not unexplored_nodes:
            return None  
        
        self.closest_node = min(unexplored_nodes.values(), key=lambda node: node.distance)'''
    
    def update_closest_node(self):
        unexplored_nodes = {node_id: node for node_id, node in self.all_nodes.items() if not node.explored}
        
        assert len(unexplored_nodes) != 0
        
        min_score = float('inf')
        closest_node = None
        
        for node_coords, node in unexplored_nodes.items():
            path = self.a_star(node_coords)
            # discount the path to node somewhat so we wont enter a never ending spiral of getting further away
            path_length = len(path) / 2
            
            if path_length > 0:  
                score = node.distance + path_length
                
                if score < min_score:
                    min_score = score
                    closest_node = node
        
        self.closest_node = closest_node

    
    def add_new_node(self, current_coords, is_explored):
        dist = distance.euclidean(current_coords, self.goal_node.coords)
        new_node = Node(coords=current_coords, explored=is_explored, distance=dist)
        self.all_nodes[current_coords] = new_node

        return new_node
    
    def generate_step(self, next_coords, curr_rotation):
        current_coords = self.current_node.coords
        x_diff = next_coords[0] - current_coords[0] 
        y_diff = next_coords[1] - current_coords[1] 
        diff = (x_diff, y_diff)

        convert = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}
        new_rotation = convert[diff]

        if new_rotation == curr_rotation:
            return {
                'action': 'move',
            }

        else:
            return {
                'action': 'rotate',
                'rotation': new_rotation,
                }

    def generate_action(self, game_state):
        #print(game_state)
        player, square = game_state['player'], game_state['square']
        current_coords = tuple(player['position'].values())

        if self.goal_node is None:
            goal_coords = tuple(game_state["target"].values())
            self.goal_node = Node(coords=goal_coords, distance=0)

        explored_nodes = {coords: node for coords, node in self.all_nodes.items() if node.explored}
        if current_coords in explored_nodes:
            self.current_node = explored_nodes[current_coords]         
        else:
            self.current_node = self.add_new_node(current_coords, is_explored=True)

        # checking for new connections and repathing incase we find a better path through 2 already discovered nodes
        self.check_for_surrounding_nodes(player, square, self.current_node)
        #self.update_all_node_connections()
        self.update_closest_node()
        self.path_to_closest = self.a_star(self.closest_node.coords)

        assert len(self.path_to_closest) != 0

        curr_rotation = player['rotation']
        next_coords = self.path_to_closest[0]

        next_move = self.generate_step(next_coords, curr_rotation)

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
        # debug log
        # print(f"Received message: {message}")  
        try:
            action, payload = json.loads(message)
            if action == 'game-instance':
                self.game_state = payload['gameState']  
                self.ready_to_start.set()
            else:
                print(f"Unhandled action type: {action}")
        except Exception as e:
            print(f"Error processing message: {e}")


    def action_loop(self):
        self.ready_to_start.wait()
        while True:
            time.sleep(0.1)
            if self.game_state:
                command = self.generate_action(json.loads(self.game_state))  
                self.ws.send(self.message('run-command', {'gameId': self.entityId, 'payload': command}))
                self.game_state = None
            

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