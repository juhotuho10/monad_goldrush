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
    coords: tuple = field(default_factory=tuple)   # (x, y)
    explored: bool = False
    distance: float = float('inf')  
    surrounding_nodes: set = field(default_factory=set)  

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
        
        self.all_nodes = {}
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
    
    def update_all_surrounding_nodes(self):
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

    def add_new_surrounding_nodes(self, player, square):
        surrounding_coords = self.get_surrounding_coords(player, square)

        surrounding_nodes = []
        for new_coords in surrounding_coords:

            if new_coords in self.all_nodes:
                new_node = self.all_nodes[new_coords]     
            else:
                dist = distance.euclidean(new_coords, self.goal_node.coords)
                new_node = Node(coords=new_coords, distance=dist)
                self.all_nodes[new_coords] = new_node
                
            surrounding_nodes.append(new_node)

        return surrounding_nodes
    
    
    def get_closest_node(self):
        unexplored_nodes = {node_id: node for node_id, node in self.all_nodes.items() if not node.explored}
        
        if not unexplored_nodes:
            return None  
        
        closest_node = min(unexplored_nodes.values(), key=lambda node: node.distance)
        
        return closest_node


    def generate_action(self, game_state):
        #print(game_state)
        player, square = game_state['player'], game_state['square']
        current_coords = tuple(player['position'].values())

        if self.goal_node is None:
            goal_coords = tuple(game_state["target"].values())
            self.goal_node = Node(coords=goal_coords, distance=0)

        if current_coords in self.all_nodes:
            self.current_node = self.all_nodes[current_coords]         
        else:
            dist = distance.euclidean(current_coords, self.goal_node.coords)
            self.current_node = Node(coords=current_coords, explored=True, distance=dist)
            self.all_nodes[current_coords] = self.current_node
            self.add_new_surrounding_nodes(player, square)
            self.update_all_surrounding_nodes()
            self.closest_node = self.get_closest_node()

        if not self.path_to_closest:
            pass
  
        rotation = player['rotation']

        walls = self.get_walls(square)

        if walls.get(rotation, False):
            possible_directions = [rot for rot, wall in walls.items() if not wall]
            new_rotation = random.choice(possible_directions) if possible_directions else 0

            return {
                'action': 'rotate',
                'rotation': new_rotation,
            }
        else:
            return {
                'action': 'move',
            }

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