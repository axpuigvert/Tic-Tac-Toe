import pygame
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. GAME ENVIRONMENT (ATOMIC ACTIONS)
# ==========================================
class TicTacToe:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.action_size = 90  # (from_cell 0..8 or 9) × (to_cell 0..8)

    def get_initial_state(self):
        return np.zeros((3, 3), dtype=np.int8)

    def get_piece_count(self, state, player):
        return np.sum(state == player)

    def get_valid_moves(self, state, player):
        valid = np.zeros(self.action_size, dtype=np.uint8)

        own_cells = np.where(state.flatten() == player)[0]
        empty_cells = np.where(state.flatten() == 0)[0]
        count = len(own_cells)

        if count < 3:
            # placement only: from_cell = 9
            for to in empty_cells:
                valid[9 * 9 + to] = 1
        else:
            # remove + place
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
            if np.all(state[i, :] == player):
                return True, 1
            if np.all(state[:, i] == player):
                return True, 1

        if state[0,0] == state[1,1] == state[2,2] == player:
            return True, 1
        if state[0,2] == state[1,1] == state[2,0] == player:
            return True, 1

        return False, 0

# ==========================================
# 2. NEURAL NETWORK (MLP)
# ==========================================
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

# ==========================================
# 3. MCTS
# ==========================================
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
                self.children.append(
                    MCTSNode(
                        self.game,
                        self.args,
                        next_state,
                        -self.player,
                        self,
                        a,
                        policy[a]
                    )
                )

    def backprop(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent:
            self.parent.backprop(-value)

class AlphaZeroAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args

    def run_mcts(self, state, player):
        root = MCTSNode(self.game, self.args, state, player)

        state_t = torch.FloatTensor((state * player).flatten()).unsqueeze(0)
        policy, _ = self.model(state_t)
        policy = policy.detach().numpy()[0]

        root.expand(policy)

        # Dirichlet noise
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
            policy, v = self.model(state_t)
            node.expand(policy.detach().numpy()[0])
            node.backprop(v.item())

        probs = np.zeros(self.game.action_size)
        for c in root.children:
            probs[c.action] = c.visit_count

        return probs / np.sum(probs)


# ==========================================
# 4. THE GUI (Updated for User Experience)
# ==========================================
class GameGUI:
    def __init__(self):
        pygame.init()
        self.width, self.height = 600, 650 # Extra space for text
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Infinity Tic-Tac-Toe (Manual Mode)")
        self.font = pygame.font.SysFont('Arial', 30)
        self.selected_remove = None
        
        # Colors
        self.BG_COLOR = (28, 170, 156)
        self.LINE_COLOR = (23, 145, 135)
        self.CIRCLE_COLOR = (239, 231, 200)
        self.CROSS_COLOR = (84, 84, 84)
        self.TEXT_COLOR = (255, 255, 255)
        
        self.game = TicTacToe()
        self.model = ZeroNet(self.game)

        # Check if the trained brain exists
        if os.path.exists("best_model.pt"):
            print("Loading Trained Brain...")
            self.model.load_state_dict(torch.load("best_model.pt"))
            self.model.eval() # Switch to "Play Mode" (freezes the weights)
        else:
            print("Warning: No model found. Playing with random brain.")

        self.args = {'C': 2.0, 'num_simulations': 1500}
        self.agent = AlphaZeroAgent(self.model, self.game, self.args)
        
        self.board = self.game.get_initial_state()
        self.player_turn = 1 # 1 = Human, -1 = AI
        self.game_over = False
        self.msg = "Your Turn"
        self.last_probs = np.zeros(9) # Store the AI's confidence distribution

    def draw_lines(self):
        self.screen.fill(self.BG_COLOR)
        # Grid
        pygame.draw.line(self.screen, self.LINE_COLOR, (0, 200), (600, 200), 15)
        pygame.draw.line(self.screen, self.LINE_COLOR, (0, 400), (600, 400), 15)
        pygame.draw.line(self.screen, self.LINE_COLOR, (200, 0), (200, 600), 15)
        pygame.draw.line(self.screen, self.LINE_COLOR, (400, 0), (400, 600), 15)
        # Status Bar
        pygame.draw.rect(self.screen, self.LINE_COLOR, (0, 600, 600, 50))
        text_surf = self.font.render(self.msg, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (20, 610))

    def draw_figures(self):
        for row in range(3):
            for col in range(3):
                # 1. DRAW THE PIECES (Existing Code)
                val = self.board[row][col]
                if val != 0:
                    age = abs(val) 
                    thickness = int(5 + (age * 8)) 
                    color = self.CROSS_COLOR if val > 0 else self.CIRCLE_COLOR
                    
                    if val > 0: # Human (X)
                        start_desc = (col * 200 + 55, row * 200 + 55)
                        end_desc = (col * 200 + 200 - 55, row * 200 + 200 - 55)
                        pygame.draw.line(self.screen, color, start_desc, end_desc, thickness)
                        start_asc = (col * 200 + 55, row * 200 + 200 - 55)
                        end_asc = (col * 200 + 200 - 55, row * 200 + 55)
                        pygame.draw.line(self.screen, color, start_asc, end_asc, thickness)
                    else: # AI (O)
                        center = (int(col * 200 + 100), int(row * 200 + 100))
                        pygame.draw.circle(self.screen, color, center, 60, thickness)

                # 2. DRAW THE "BRAIN SCAN" (New Code)
                # Show the probability the AI assigned to this square
                idx = row * 3 + col
                if self.last_probs[idx] > 0:
                    # Render the probability as a small percentage (e.g., "45%")
                    prob_text = f"{int(self.last_probs[idx] * 100)}%"
                    
                    # Choose color: Green if high confidence, Grey if low
                    text_color = (0, 100, 0) if self.last_probs[idx] > 0.2 else (150, 150, 150)
                    
                    # Draw text in the corner of the square
                    prob_surf = self.font.render(prob_text, True, text_color)
                    self.screen.blit(prob_surf, (col * 200 + 5, row * 200 + 5))

    def run(self):
        self.draw_lines()
        
        while True:
            # DYNAMIC MESSAGE UPDATE
            # Check if Human needs to remove or place
            if not self.game_over and self.player_turn == 1:
                p1_count = np.sum(self.board == 1)
                if p1_count == 3:
                    self.msg = "MAX PIECES! Select an X to remove."
                else:
                    self.msg = "Your Turn (Place X)"
            elif not self.game_over:
                self.msg = "AI is thinking..."
                
            self.draw_lines()
            self.draw_figures()
            pygame.display.update()

            # EVENT LOOP
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                # --- HUMAN TURN ---
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over and self.player_turn == 1:
                    mouseX, mouseY = event.pos
                    if mouseY > 600:
                        continue

                    row = mouseY // 200
                    col = mouseX // 200
                    cell = int(row * 3 + col)

                    valid_moves = self.game.get_valid_moves(self.board, 1)
                    piece_count = np.sum(self.board == 1)

                    # -------------------------------
                    # PLACEMENT PHASE (< 3 pieces)
                    # -------------------------------
                    if piece_count < 3:
                        action = 9 * 9 + cell
                        if valid_moves[action]:
                            self.board = self.game.get_next_state(self.board, action, 1)
                            is_win, _ = self.game.check_win(self.board, 1)
                            if is_win:
                                self.msg = "YOU WIN!"
                                self.game_over = True
                            else:
                                self.player_turn = -1
                            self.last_probs = np.zeros(9)

                    # -------------------------------
                    # REMOVE + PLACE PHASE
                    # -------------------------------
                    else:
                        # Step 1: select piece to remove
                        if self.selected_remove is None:
                            if self.board[row][col] == 1:
                                self.selected_remove = cell
                                self.msg = "Select where to place X"
                        # Step 2: select destination
                        else:
                            action = self.selected_remove * 9 + cell
                            if valid_moves[action]:
                                self.board = self.game.get_next_state(self.board, action, 1)
                                self.selected_remove = None
                                is_win, _ = self.game.check_win(self.board, 1)
                                if is_win:
                                    self.msg = "YOU WIN!"
                                    self.game_over = True
                                else:
                                    self.player_turn = -1
                                self.last_probs = np.zeros(9)


            # --- AI TURN ---
            if self.player_turn == -1 and not self.game_over:
                # 1. AI Decision
                mcts_probs = self.agent.run_mcts(self.board, -1)
                self.last_probs = mcts_probs
                best_move = np.argmax(mcts_probs)
                
                # 2. Apply Move
                prev_board = self.board.copy()
                self.board = self.game.get_next_state(self.board, best_move, -1)
                
                # 3. Check Logic: Remove or Place?
                pieces_before = np.sum(prev_board == -1)
                pieces_after = np.sum(self.board == -1)
                
                if pieces_after < pieces_before:
                    # AI just removed a piece. It goes AGAIN immediately.
                    time.sleep(0.5) # Slight pause so human sees the removal
                    continue 
                else:
                    # AI placed a piece. Check Win.
                    is_win, _ = self.game.check_win(self.board, -1)
                    if is_win:
                        self.msg = "AI WINS!"
                        self.game_over = True
                    else:
                        self.player_turn = 1

if __name__ == "__main__":
    gui = GameGUI()
    gui.run()