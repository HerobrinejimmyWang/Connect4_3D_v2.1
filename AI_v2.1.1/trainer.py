import os
import time
import copy
import logging
import torch
import torch.optim as optim
import numpy as np
import multiprocessing
from tqdm import tqdm
from collections import deque
from random import shuffle

from game_rules import GameRules
from model import Connect4Net
from mcts import MCTS

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainerArgs:
    def __init__(self):
        self.num_iterations = 200     # Total training iterations
        self.num_self_play_games = 100 # Games per iteration (Parallelized)
        self.num_mcts_sims = 64       # MCTS simulations per move
        self.cpuct = 1.0              # PUCT exploration constant
        self.batch_size = 64          # Training batch size
        self.epochs = 10              # Training epochs per iteration
        self.checkpoint_dir = './checkpoints'
        self.learning_rate = 0.001
        self.history_len = 5          # Number of iterations to keep history
        self.checkpoint_interval = 5  # Checkpoint every X iterations

# --- Worker Function for Parallel Processing ---
# Must be top-level for pickling in multiprocessing
def self_play_worker(args_tuple):
    """
    Worker function to play ONE game independently.
    Args: (game_rules, model_state_dict, trainer_args, seed)
    """
    game_rules, model_state, args, seed = args_tuple
    
    # Set seed for this worker
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reconstruct environment
    game = GameRules()
    net = Connect4Net()
    net.load_state_dict(model_state)
    net.eval() # Set to eval mode for inference
    
    mcts = MCTS(game, net, args)
    
    train_examples = []
    board = game.get_init_board()
    cur_player = 1
    episode_step = 0
    
    while True:
        episode_step += 1
        canonical_board = game.get_canonical_form(board, cur_player)
        temp = int(episode_step < 15) # Temperature high early in game
        
        pi = mcts.get_action_prob(canonical_board, temp=temp)
        
        # Symmetries (optional but recommended for 3D board to augment data)
        # For simplicity, we just store raw data here. 
        # Adding rotation/flipping for 3D is complex, omitting for v1.
        train_examples.append([canonical_board, cur_player, pi, None])
        
        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.get_next_state(board, cur_player, action)
        
        r = game.get_game_ended(board, cur_player)
        
        if r != 0:
            # Game ended. r is from perspective of cur_player
            # Backfill rewards
            return_data = []
            for x in train_examples:
                # x[1] is the player who made the move resulting in x[0] state
                # if x[1] == cur_player, reward is r * ((-1) if r != 1e-4) -> Logic check:
                # Actually simpler: r is reward for cur_player.
                # If step player was cur_player, reward is r. If step player was -cur_player, reward is -r.
                reward = r * (1 if x[1] == cur_player else -1)
                return_data.append((x[0], x[2], reward))
            return return_data

class Trainer:
    def __init__(self, args, resume_path=None):
        self.args = args
        self.game = GameRules()
        self.nnet = Connect4Net()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.learning_rate)
        
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            
        self.train_examples_history = []  # history of examples
        self.start_iter = 1

        # 新增：保存评估历史
        self.eval_history = []  # list of dicts: {'iteration', 'wins','losses','draws','games'}
        
        # Resume functionality
        if resume_path and os.path.isfile(resume_path):
            logging.info(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_iter = checkpoint['iteration'] + 1
            logging.info(f"Resuming from iteration {self.start_iter}")
        else:
            logging.info("Starting training from scratch.")

    def execute_episode_parallel(self):
        """
        Runs self-play using multiprocessing pool.
        """
        cpu_count = multiprocessing.cpu_count()
        # Use 50% of CPU cores, ensuring at least 1
        num_workers = max(1, int(cpu_count * 0.5))
        logging.info(f"Spawning {num_workers} workers for self-play...")
        
        model_state = self.nnet.state_dict()
        
        # Prepare arguments for each game
        # We need unique seeds for randomness
        tasks = []
        for i in range(self.args.num_self_play_games):
            seed = int(time.time()) + i
            tasks.append((self.game, model_state, self.args, seed))
        
        iteration_examples = []
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # tqdm for progress bar
            results = list(tqdm(pool.imap(self_play_worker, tasks), total=self.args.num_self_play_games, desc="Self-Play"))
            
            for game_data in results:
                iteration_examples.extend(game_data)
                
        return iteration_examples

    def train(self):
        for i in range(self.start_iter, self.args.num_iterations + 1):
            logging.info(f'Starting Iteration {i}/{self.args.num_iterations}')
            
            # 1. Self-Play
            iter_examples = self.execute_episode_parallel()
            self.train_examples_history.append(iter_examples)
            
            # Keep history limited
            if len(self.train_examples_history) > self.args.history_len:
                logging.info(f"Removing oldest history (keep last {self.args.history_len})")
                self.train_examples_history.pop(0)
            
            # Flatten list
            train_data = []
            for e in self.train_examples_history:
                train_data.extend(e)
            shuffle(train_data)
            
            # 2. Train Neural Net
            self.train_network(train_data)
            
            # 3. Save Checkpoint & Evaluate
            if i % self.args.checkpoint_interval == 0:
                self.save_checkpoint(i)
                self.evaluate_model(i)

        # 训练循环结束后：确保最终模型被保存一次
        logging.info("Saving final checkpoint...")
        self.save_checkpoint(self.args.num_iterations)
        self.final_report()

    def train_network(self, examples):
        """
        examples: list of (board, policy, value)
        """
        self.nnet.train()
        batch_count = int(len(examples) / self.args.batch_size)
        
        pbar = tqdm(range(batch_count), desc="Training Net")
        total_loss = 0
        
        for _ in pbar:
            sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
            boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            
            boards = torch.FloatTensor(np.array(boards).astype(np.float64))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
            
            # Predict
            out_pi, out_v = self.nnet(boards)
            
            # Loss: value_loss + policy_loss
            # value loss = mean squared error
            # policy loss = cross entropy (log_softmax output already)
            l_v = F.mse_loss(out_v.view(-1), target_vs.view(-1))
            l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
            
            total_l = l_v + l_pi
            
            self.optimizer.zero_grad()
            total_l.backward()
            self.optimizer.step()
            
            total_loss += total_l.item()
            pbar.set_postfix(loss=total_loss/(_ + 1))

    def save_checkpoint(self, iteration):
        """
        仅保存模型本身（state_dict），以文件夹形式： checkpoints/checkpoint_{iteration}/model.pth
        """
        dirpath = os.path.join(self.args.checkpoint_dir, f'checkpoint_{iteration}')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        model_path = os.path.join(dirpath, 'model.pth')
        # 仅保存模型 state_dict
        torch.save(self.nnet.state_dict(), model_path)
        logging.info(f"Model saved: {model_path}")

    def evaluate_model(self, iteration):
        """
        Simple evaluation: Play against a random agent (or greedy) to check progress.
        """
        logging.info(f"--- Evaluating model at iteration {iteration} ---")
        # Let's play 10 games against a Random Agent
        # Random Agent logic:
        
        wins = 0
        losses = 0
        draws = 0
        
        # We use a single thread for evaluation to save overhead
        eval_game = GameRules()
        mcts = MCTS(eval_game, self.nnet, self.args)
        
        for _ in range(10): # Play 10 games
            board = eval_game.get_init_board()
            cur_player = 1
            game_over = False
            
            # Model plays as Player 1
            while not game_over:
                if cur_player == 1:
                    # Model Move
                    canonical_board = eval_game.get_canonical_form(board, cur_player)
                    pi = mcts.get_action_prob(canonical_board, temp=0) # Deterministic
                    action = np.argmax(pi)
                else:
                    # Random Opponent Move
                    valid_moves = eval_game.get_valid_moves(board)
                    valid_indices = np.where(valid_moves == 1)[0]
                    if len(valid_indices) > 0:
                        action = np.random.choice(valid_indices)
                    else:
                        action = 0 # Should not happen unless draw
                        
                board, cur_player = eval_game.get_next_state(board, cur_player, action)
                r = eval_game.get_game_ended(board, 1) # Check from P1 perspective
                
                if r != 0:
                    if r == 1: wins += 1
                    elif r == -1: losses += 1
                    else: draws += 1
                    game_over = True
                    
        logging.info(f"Eval Result (vs Random) - Wins: {wins}, Losses: {losses}, Draws: {draws}")
        with open(os.path.join(self.args.checkpoint_dir, "eval_log.txt"), "a") as f:
            f.write(f"Iter {iteration}: Wins {wins}/10, Losses {losses}, Draws {draws}\n")

        # 新增：记录到内存中的评估历史，便于最终报告汇总
        self.eval_history.append({
            'iteration': iteration,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'games': 10
        })

    def final_report(self):
        report_path = os.path.join(self.args.checkpoint_dir, "FINAL_REPORT.txt")
        with open(report_path, "w") as f:
            f.write("=== Training Final Report ===\n")
            f.write(f"Total Iterations: {self.args.num_iterations}\n")
            f.write(f"Self-Play Games per Iter: {self.args.num_self_play_games}\n")
            f.write("Model trained successfully.\n")
            f.write("Check 'eval_log.txt' for progress history.\n")

            # 新增：如果有评估历史，输出评估指标摘要
            if len(self.eval_history) > 0:
                f.write("\n=== Evaluation Summary ===\n")
                total_evals = len(self.eval_history)
                f.write(f"Number of Evaluations: {total_evals}\n")
                # 计算平均/最佳/最后胜率
                win_rates = [e['wins']/e['games'] for e in self.eval_history]
                avg_win = sum(win_rates)/len(win_rates)
                best_idx = int(np.argmax(win_rates))
                best_entry = self.eval_history[best_idx]
                last_entry = self.eval_history[-1]
                f.write(f"Average Win Rate (vs Random): {avg_win:.3f}\n")
                f.write(f"Best Win Rate: {win_rates[best_idx]:.3f} at Iter {best_entry['iteration']} (W/D/L = {best_entry['wins']}/{best_entry['draws']}/{best_entry['losses']})\n")
                f.write(f"Last Eval (Iter {last_entry['iteration']}): Wins {last_entry['wins']}, Losses {last_entry['losses']}, Draws {last_entry['draws']}\n")
            
            # 最终模型路径信息
            final_model_path = os.path.join(self.args.checkpoint_dir, 'final.pth.tar')
            if os.path.exists(final_model_path):
                f.write(f"\nFinal model saved at: {final_model_path}\n")
            else:
                f.write("\nFinal model file not found. Checkpoints available in the checkpoint directory.\n")
        logging.info(f"Training Complete. Report saved to {report_path}")

import torch.nn.functional as F # Import needed for loss function