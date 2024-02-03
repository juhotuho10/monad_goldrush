import os
import json
import random
import webbrowser
import requests
import websocket
from dotenv import load_dotenv
import threading
import time

load_dotenv()

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

    def generate_action(self, game_state):
        player, square = game_state['player'], game_state['square']
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
        print(f"Received message: {message}")  
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
            if self.game_state:
                command = self.generate_action(json.loads(self.game_state))  
                self.ws.send(self.message('run-command', {'gameId': self.entityId, 'payload': command}))
                self.game_state = None
            time.sleep(0.1)

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