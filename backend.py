from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

app = Flask(__name__)
CORS(app)  # Permet que React parli amb aquest servidor

# ==========================================
# 1. CLASSES DEL JOC (Copiades del teu TicTacToe.py)
# ==========================================
class TicTacToe:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.action_size = 90

    def get_valid_moves(self, state, player):
        valid = np.zeros(self.action_size, dtype=np.uint8)
        own_cells = np.where(state.flatten() == player)[0]
        empty_cells = np.where(state.flatten() == 0)[0]
        count = len(own_cells)

        if count < 3:
            for to in empty_cells:
                valid[9 * 9 + to] = 1
        else:
            for frm in own_cells:
                for to in empty_cells:
                    valid[frm * 9 + to] = 1
        return valid

    def get_next_state(self, state, action, player):
        frm = action // 9
        to = action % 9
        new_state = state.copy()
        if frm != 9:
            r, c = divmod(frm, 3)
            new_state[r, c] = 0
        r, c = divmod(to, 3)
        new_state[r, c] = player
        return new_state

    def check_win(self, state, player):
        for i in range(3):
            if np.all(state[i, :] == player): return True, 1
            if np.all(state[:, i] == player): return True, 1
        if state[0,0] == state[1,1] == state[2,2] == player: return True, 1
        if state[0,2] == state[1,1] == state[2,0] == player: return True, 1
        return False, 0

class ZeroNet(nn.Module):
    def __init__(self, game):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, game.action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

class MCTSNode:
    def __init__(self, game, args, state, player, parent=None, action=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.value_sum = 0

    def is_expanded(self):
        return len(self.children) > 0

    def ucb(self, child):
        q = 0 if child.visit_count == 0 else child.value_sum / child.visit_count
        u = self.args['C'] * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return q + u

    def select(self):
        return max(self.children, key=lambda c: self.ucb(c))

    def expand(self, policy):
        valid = self.game.get_valid_moves(self.state, self.player)
        for a in range(self.game.action_size):
            if valid[a]:
                next_state = self.game.get_next_state(self.state, a, self.player)
                self.children.append(MCTSNode(self.game, self.args, next_state, -self.player, self, a, policy[a]))

    def backprop(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent: self.parent.backprop(-value)

class AlphaZeroAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args

    def run_mcts(self, state, player):
        root = MCTSNode(self.game, self.args, state, player)
        state_t = torch.FloatTensor((state * player).flatten()).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = self.model(state_t)
            policy = policy.detach().numpy()[0]

        root.expand(policy)

        # Soroll per variabilitat
        noise = np.random.dirichlet([0.3] * len(root.children))
        for i, c in enumerate(root.children):
            c.prior = 0.75 * c.prior + 0.25 * noise[i]

        for _ in range(self.args['num_simulations']):
            node = root
            while node.is_expanded():
                node = node.select()

            is_win, val = self.game.check_win(node.state, -node.player)
            if is_win:
                node.backprop(val)
                continue

            state_t = torch.FloatTensor((node.state * node.player).flatten()).unsqueeze(0)
            with torch.no_grad():
                policy, v = self.model(state_t)
            node.expand(policy.detach().numpy()[0])
            node.backprop(v.item())

        probs = np.zeros(self.game.action_size)
        for c in root.children:
            probs[c.action] = c.visit_count
        
        sum_probs = np.sum(probs)
        return probs / sum_probs if sum_probs > 0 else probs

# ==========================================
# 2. INICIALITZACIÓ GLOBAL
# ==========================================
game = TicTacToe()
model = ZeroNet(game)
loaded_model = False

if os.path.exists("best_model.pt"):
    try:
        model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
        model.eval()
        loaded_model = True
        print("✅ Model 'best_model.pt' carregat correctament!")
    except Exception as e:
        print(f"❌ Error carregant el model: {e}")
else:
    print("⚠️ 'best_model.pt' no trobat. Jugant amb pesos aleatoris.")

# ==========================================
# 3. ENDPOINTS DE L'API
# ==========================================
@app.route('/api/move', methods=['POST'])
def get_move():
    data = request.json
    board_list = data.get('board') # Llista plana de 9 elements (1, -1, 0)
    difficulty = data.get('difficulty', 'Difícil')
    
    # Convertir tauler de React (null, 'X', 'O') a format numpy (0, 1, -1)
    # Assumint que la IA sempre juga com a -1 ('O') i l'Humà com a 1 ('X')
    # Però el model espera que '1' sigui el jugador actual. 
    # Així que invertirem el tauler perquè la IA es vegi a si mateixa com a '1'.
    
    state = np.zeros((3, 3), dtype=np.int8)
    for i, val in enumerate(board_list):
        r, c = divmod(i, 3)
        if val == 'X': state[r, c] = 1   # Humà
        elif val == 'O': state[r, c] = -1 # IA

    # Determinar simulacions segons dificultat
    sims = 200
    if difficulty == 'Fàcil': sims = 0
    elif difficulty == 'Mitjà': sims = 20
    elif difficulty == 'Difícil': sims = 50
    elif difficulty == 'Extrem': sims = 800

    if sims == 0:
        # Random move
        valid = game.get_valid_moves(state, -1)
        valid_indices = np.where(valid == 1)[0]
        best_move = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else None
    else:
        # AlphaZero MCTS
        # La IA és el jugador -1. Però el model està entrenat per jugar com a '1'.
        # Passem el tauler tal qual, però li diem a l'agent que jugui com a -1.
        # Dins de run_mcts, multipliquem state * player per normalitzar la vista.
        
        args = {'C': 1.4, 'num_simulations': sims}
        agent = AlphaZeroAgent(model, game, args)
        probs = agent.run_mcts(state, -1)
        best_move = int(np.argmax(probs))

    # Convertir acció (0-89) a format comprensible per React (type, from, to)
    # React espera l'acció aplicada, o podem retornar l'índex atomic
    # Retornem l'índex atomic i deixem que React actualitzi
    
    return jsonify({'action': best_move})

if __name__ == '__main__':
    app.run(debug=True, port=5000)