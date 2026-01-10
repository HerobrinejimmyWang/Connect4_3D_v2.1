import sys
import os
import torch
import numpy as np
import glob
import re
import importlib.util

# Add AI folder to path so we can import its modules
current_dir = os.path.dirname(os.path.abspath(__file__))

# 自动发现以 "AI_" 开头的文件夹
def get_ai_generations():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [d for d in os.listdir(current_dir)
                  if os.path.isdir(os.path.join(current_dir, d)) and d.startswith('AI_')]
    return sorted(candidates)

candidates = get_ai_generations()
if not candidates:
    raise RuntimeError("未找到 AI_* 文件夹，请检查路径或手动设置 ai_path")
ai_path = os.path.join(current_dir, candidates[0])

sys.path.append(ai_path)

def _load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find module file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class AI_Interface:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_ai_path = ai_path
        self._load_ai_modules(self.current_ai_path)
        self.dims = (8, 5, 5) # Layer, Row, Col
        self.loaded_model_path = None
        self.nnet = None
        self.args = type('Args', (), {})()
        self.args.num_mcts_sims = 100 # Default simulation count for gameplay
        self.args.cpuct = 1.0

    def _load_ai_modules(self, path):
        """动态加载指定路径下的 AI 模块"""
        try:
            if path not in sys.path:
                sys.path.append(path)
            
            self.gr_mod = _load_module_from_file('game_rules', os.path.join(path, 'game_rules.py'))
            self.model_mod = _load_module_from_file('model', os.path.join(path, 'model.py'))
            self.mcts_mod = _load_module_from_file('mcts', os.path.join(path, 'mcts.py'))

            self.game = self.gr_mod.GameRules()
            self.Connect4Net = self.model_mod.Connect4Net
            self.MCTS = self.mcts_mod.MCTS
            self.current_ai_path = path
            print(f"AI Modules loaded from: {path}")
        except Exception as e:
            print(f"Error loading AI modules from {path}: {e}")

    def switch_generation(self, folder_name):
        """切换到不同版本的 AI 文件夹"""
        new_path = os.path.join(current_dir, folder_name)
        if os.path.exists(new_path):
            self._load_ai_modules(new_path)
            return True
        return False

    def find_checkpoints(self, target_path=None):
        """搜索目标 AI 目录下的所有 checkpoint，包括 save_model 和 checkpoints 文件夹"""
        base_path = target_path if target_path else self.current_ai_path
        search_dirs = [os.path.join(base_path, 'checkpoints'), os.path.join(base_path, 'save_model')]
        
        checkpoints = []
        pattern = re.compile(r'checkpoint_(\d+)')
        
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            
            for root, dirs, files in os.walk(s_dir):
                for file in files:
                    # 匹配常见的模型后缀
                    if file == 'model.pth' or file.endswith(('.pt', '.pth', '.ckpt')):
                        parent_folder = os.path.basename(root)
                        match = pattern.search(parent_folder) or pattern.search(file)
                        iter_num = int(match.group(1)) if match else 0
                        
                        full_path = os.path.join(root, file)
                        # 构造显示名称，包含它所属的子文件夹（如 save_model）
                        tag = " (Latest)" if 'save_model' in root else ""
                        display_name = f"{parent_folder}{tag}"
                        
                        checkpoints.append({
                            'path': full_path, 
                            'iter': iter_num, 
                            'name': display_name,
                            'type': 'save_model' if 'save_model' in root else 'checkpoint'
                        })
        
        # 优先排序 save_model 中的，然后按迭代次数降序
        checkpoints.sort(key=lambda x: (x['type'] != 'save_model', -x['iter']))
        return checkpoints

    def load_model(self, model_path):
        """Loads a specific model checkpoint."""
        try:
            self.nnet = self.Connect4Net()
            # Handle Map Location (CPU/CUDA)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Use strict=False to ignore keys that might be slightly off if versions mismatch, 
            # though they should match based on repo structure.
            # Usually checkpoint is state_dict directly or dict with state_dict
            if 'state_dict' in checkpoint:
                self.nnet.load_state_dict(checkpoint['state_dict'])
            else:
                self.nnet.load_state_dict(checkpoint)
                
            self.nnet.to(self.device)
            self.nnet.eval()
            self.loaded_model_path = model_path
            print(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def get_game_board(self, board_array):
        """Converts game board (0,1,2) to AI board (0,1,-1)."""
        # Game: 0=Empty, 1=Red, 2=Blue
        # AI: 1=Player, -1=Opponent.
        # Ensure we always treat "Current Player" as 1 for the canonical form logic
        # But GameRules expects raw board with 1 and -1.
        ai_board = np.copy(board_array)
        ai_board[ai_board == 2] = -1
        return ai_board

    def get_prediction(self, board_array, current_player):
        """
        Returns win rate prediction for the current player (0.0 to 1.0).
        Includes 0% - 100%.
        """
        if not self.nnet:
            return 0.5

        # Prepare board
        ai_board = self.get_game_board(board_array) 
        
        # GameRules.get_canonical_form(board, player) -> returns board from player perspective
        # If player is 2 (-1), it flips the signs so the model sees '1' as itself.
        player_val = 1 if current_player == 1 else -1
        canonical_board = self.game.get_canonical_form(ai_board, player_val)
        
        # Run through network
        board_tensor = torch.FloatTensor(canonical_board.astype(np.float64))
        board_tensor = board_tensor.to(self.device)
        
        with torch.no_grad():
            _, v = self.nnet(board_tensor.unsqueeze(0))
        
        # v is in [-1, 1]. -1 means loss, 1 means win (for canonical player).
        # Convert to 0-1 range probability
        win_prob = (v.item() + 1) / 2
        return win_prob

    def get_best_move(self, board_array, current_player, timeout=2.0):
        """
        Runs MCTS to find the best move.
        Returns (layer, row, col).
        """
        if not self.nnet:
            return None

        ai_board = self.get_game_board(board_array)
        player_val = 1 if current_player == 1 else -1
        canonical_board = self.game.get_canonical_form(ai_board, player_val)
        
        mcts = self.MCTS(self.game, self.nnet, self.args)
        
        # Determine number of sims based on performance? 
        # Fixed count is better for consistent "thinking" time logic, usually.
        # But MCTS class here runs sims in a batch loop or one by one? 
        # mcts.get_action_prob runs internal loop.
        
        # MCTS works on canonical board
        probs = mcts.get_action_prob(canonical_board, temp=0) # temp=0 for best move (argmax)
        action = np.argmax(probs)
        
        # Convert simple integer action back to (layer, row, col)
        # Action index = layer * 25 + row * 5 + col
        # Board dimensions are AI specific (8, 5, 5)
        
        layer = action // 25
        rem = action % 25
        row = rem // 5
        col = rem % 5
        
        return (layer, row, col)

    def is_valid_move(self, board_array, layer, row, col):
        ai_board = self.get_game_board(board_array)
        # Manually check validity using game logic pattern, or use GameRules
        # GameRules.get_valid_moves works on the whole board.
        if board_array[layer, row, col] != 0:
            return False
            
        # Gravity
        if layer == 0:
            return True
        if board_array[layer-1, row, col] != 0:
            return True
            
        return False
