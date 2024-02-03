import os
import json
import random
import webbrowser
import requests
import websocket
from dotenv import load_dotenv

load_dotenv()

# Loading variables
PLAYER_TOKEN = os.getenv('PLAYER_TOKEN')
LEVEL_ID = os.getenv('LEVEL_ID')
FRONTEND_BASE = 'goldrush.monad.fi'
BACKEND_BASE = 'goldrush.monad.fi/backend'

def message(action, payload=None):
    return json.dumps([action, payload or {}])

def get_walls(square):
    masks = [0b1000, 0b0100, 0b0010, 0b0001]
    return {
        0: (square & masks[0]) != 0,
        90: (square & masks[1]) != 0,
        180: (square & masks[2]) != 0,
        270: (square & masks[3]) != 0,
    }

def generate_action(game_state):
    player, square = game_state['player'], game_state['square']
    rotation = player['rotation']

    walls = get_walls(square)

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

def create_game(level_id, token):
    url = f'https://{BACKEND_BASE}/api/levels/{level_id}'
    headers = {'Authorization': token}
    response = requests.post(url, headers=headers)

    if not response.ok:
        print(f"Couldn't create game: {response.status_code} - {response.text}")
        return None

    return response.json()

def on_message(ws, message):
    action, payload = json.loads(message)
    if action != 'game-instance':
        print([action, payload])
        return

    game_state = json.loads(payload['gameState'])
    commands = generate_action(game_state)
    print(commands)
    ws.send(message('run-command', {'gameId': game_state['entityId'], 'payload': commands}))


def main():
    game = create_game(LEVEL_ID, PLAYER_TOKEN)
    if not game:
        return

    url = f'https://{FRONTEND_BASE}/?id={game["entityId"]}'
    print(f'Game at {url}')
    webbrowser.open(url)

    ws_url = f'wss://{BACKEND_BASE}/{PLAYER_TOKEN}/'
    ws = websocket.WebSocketApp(ws_url,on_message=on_message)
    ws.on_open = lambda ws: ws.send(message('sub-game', {'id': game['entityId']}))
    ws.run_forever()

if __name__ == "__main__":
    main()
