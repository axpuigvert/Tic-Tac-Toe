import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from TicTacToe import TicTacToe, ZeroNet, AlphaZeroAgent

class Trainer:
    def __init__(self):
        self.args = {
            'num_iterations': 500,
            'num_self_play_games': 100,
            'epochs': 4,
            'batch_size': 64,
            'lr': 0.001,
            'C': 1.4,
            'num_simulations': 200
        }

        self.game = TicTacToe()
        self.model = ZeroNet(self.game)

        if os.path.exists("best_model.pt"):
            self.model.load_state_dict(torch.load("best_model.pt"))
            print("Loaded existing model")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        self.agent = AlphaZeroAgent(self.model, self.game, self.args)

    def execute_episode(self):
        memory = []
        state = self.game.get_initial_state()
        player = 1
        step = 0

        while True:
            step += 1
            probs = self.agent.run_mcts(state, player)
            memory.append((state, probs, player))

            action = np.random.choice(len(probs), p=probs) if step < 10 else np.argmax(probs)
            state = self.game.get_next_state(state, action, player)

            win, _ = self.game.check_win(state, player)
            if win:
                return [(s, p, 1 if pl == player else -1) for s, p, pl in memory]

            if step > 100:
                return [(s, p, 0) for s, p, _ in memory]

            player = -player

    def train(self, data):
        np.random.shuffle(data)

        for epoch in range(self.args['epochs']):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            batches = 0

            for i in range(0, len(data), self.args['batch_size']):
                batch = data[i:i+self.args['batch_size']]
                if len(batch) == 0:
                    continue

                boards = torch.FloatTensor([b[0].flatten() for b in batch])
                target_pis = torch.FloatTensor([b[1] for b in batch])
                target_vs = torch.FloatTensor([b[2] for b in batch])

                out_pi, out_v = self.model(boards)

                # ---- LOSSES ----
                loss_v = F.mse_loss(out_v.view(-1), target_vs)
                loss_pi = -torch.sum(
                    target_pis * torch.log(out_pi + 1e-8)
                ) / len(batch)

                loss = loss_v + loss_pi
                # ----------------

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += loss_pi.item()
                total_value_loss += loss_v.item()
                batches += 1

            print(
                f"Epoch {epoch+1}: "
                f"Total={total_loss/batches:.4f} | "
                f"Policy={total_policy_loss/batches:.4f} | "
                f"Value={total_value_loss/batches:.4f}"
            )

    def learn(self):
        for it in range(self.args['num_iterations']):
            print(f"Iteration {it+1}")
            data = []
            for _ in range(self.args['num_self_play_games']):
                data += self.execute_episode()

            self.train(data)
            torch.save(self.model.state_dict(), "best_model.pt")
            print("Saved model")

if __name__ == "__main__":
    Trainer().learn()
