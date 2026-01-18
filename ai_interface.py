import sys
import os
import torch
import numpy as np
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
        self.mcts = None # 预加载的 MCTS 对象
        self.args = type('Args', (), {})()
        self.args.num_mcts_sims = 100 # Default simulation count for gameplay
        self.args.cpuct = 1.0

    def _load_ai_modules(self, path):
        """动态加载指定路径下的 AI 模块"""
        try:
            # 清理旧的 AI 路径并添加新路径到开头，确保模块查找优先级
            for p in sys.path[:]:
                if "AI_v" in p:
                    sys.path.remove(p)
            sys.path.insert(0, path)
            
            self.gr_mod = _load_module_from_file('game_rules', os.path.join(path, 'game_rules.py'))
            self.model_mod = _load_module_from_file('model', os.path.join(path, 'model.py'))
            self.mcts_mod = _load_module_from_file('mcts', os.path.join(path, 'mcts.py'))

            self.game = self.gr_mod.GameRules()
            self.Connect4Net = self.model_mod.Connect4Net
            self.MCTS = self.mcts_mod.MCTS
            self.current_ai_path = path
            self.mcts = None # 模块更换时清空 MCTS
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
        """搜索目标 AI 目录下的 save_model 文件夹，按分级读取模型"""
        base_path = target_path if target_path else self.current_ai_path
        save_model_dir = os.path.join(base_path, 'save_model')
        
        checkpoints = []
        if not os.path.exists(save_model_dir):
            print(f"Warning: {save_model_dir} not found.")
            return checkpoints
        
        # 优先级映射：Low | Middle | High
        difficulty_map = {"High": 0, "Middle": 1, "Low": 2}
        
        # 遍历 save_model 目录
        items = os.listdir(save_model_dir)
        for item in items:
            item_path = os.path.join(save_model_dir, item)
            
            # 1. 检查子文件夹形式（如 save_model/Low/model.pth）
            if os.path.isdir(item_path):
                # 寻找文件夹内的第一个模型文件
                sub_files = os.listdir(item_path)
                for file in sub_files:
                    if file.endswith(('.pt', '.pth', '.ckpt', '.pth.tar')):
                        full_path = os.path.join(item_path, file)
                        checkpoints.append({
                            'path': full_path, 
                            'name': item,  # 使用文件夹名作为显示名称
                            'rank': difficulty_map.get(item, 99)
                        })
                        break # 一个文件夹只取一个模型
            
            # 2. 检查直接放在 save_model 根目录下的模型（兼容性）
            elif item.endswith(('.pt', '.pth', '.ckpt', '.pth.tar')):
                # 去掉后缀作为显示名称
                display_name = item.split('.')[0]
                checkpoints.append({
                    'path': item_path,
                    'name': display_name,
                    'rank': difficulty_map.get(display_name, 99)
                })

        # 按优先级排序 (High -> Middle -> Low)
        checkpoints.sort(key=lambda x: x['rank'])
        
        # 打印发现的模型，方便调试
        if checkpoints:
            print(f"Discovered {len(checkpoints)} models in {save_model_dir}:")
            for cp in checkpoints:
                print(f"  - {cp['name']} ({cp['path']})")
        else:
            print(f"No models found in {save_model_dir}")
            
        return checkpoints

    def load_model(self, model_path, rank=None):
        """加载指定的模型 Checkpoint，具有兼容不同保存方式的强健性"""
        try:
            # 根据模型等级调整搜索次数，High 加深思考，Low 减少负担
            if rank == 0: # High
                self.args.num_mcts_sims = 100
            elif rank == 1: # Middle
                self.args.num_mcts_sims = 64
            elif rank == 2: # Low
                self.args.num_mcts_sims = 64
            else:
                self.args.num_mcts_sims = 64

            # 重新实例化网络，确保使用当前加载的模块架构
            self.nnet = self.Connect4Net()
            
            # 兼容性加载：处理 weights_only 限制和不同的序列化对象
            checkpoint = None
            try:
                # 首先尝试安全的 weights_only 加载 (PyTorch 推荐)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                # 如果失败（例如包含 numpy 对象或自定义类），回退到完整加载
                print(f"Warning: secure load failed for {os.path.basename(model_path)}, falling back.")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 提取 state_dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # 假设字典本身就是 state_dict
                    state_dict = checkpoint
            else:
                # 兼容直接保存模型对象的旧方式
                state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                
            self.nnet.load_state_dict(state_dict)
            self.nnet.to(self.device)
            self.nnet.eval()
            self.loaded_model_path = model_path
            
            # 切换模型时强制重置 MCTS
            self.mcts = self.MCTS(self.game, self.nnet, self.args)
            
            print(f"Model successfully loaded: {model_path} (Sims: {self.args.num_mcts_sims})")
            return True
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
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

    def get_best_move(self, board_array, current_player, timeout=2.0, step_count=0):
        """
        Runs MCTS to find the best move with optimizations.
        Returns (layer, row, col).
        """
        if not self.nnet:
            return None

        ai_board = self.get_game_board(board_array)
        player_val = 1 if current_player == 1 else -1
        canonical_board = self.game.get_canonical_form(ai_board, player_val)
        
        # --- 优化 1: 立即获胜或阻止对方获胜检测 ---
        # 这种硬编码检查可以节省大量 MCTS 时间
        immediate_move = self._find_immediate_win_or_block(canonical_board)
        if immediate_move is not None:
            print(f"Immediate move found: action {immediate_move}")
            return self._action_to_coords(immediate_move)

        # --- 优化 2: 开局减少 MCTS 搜索次数 ---
        # 暂存原始搜索量
        original_sims = self.args.num_mcts_sims
        if step_count < 12:
            # 开局前 12 步，降低为 30% 搜索量，最小不低于 20
            self.args.num_mcts_sims = max(20, int(original_sims * 0.3))

        # 复用 MCTS 对象，保留之前的搜索树以提升“智商”
        if self.mcts is None:
            self.mcts = self.MCTS(self.game, self.nnet, self.args)
        
        # --- 优化 3: 分阶段忽略高层内容 ---
        # 通过修改有效动作掩码 (Valid Moves Mask) 来限制搜索空间
        valid_moves = self.game.get_valid_moves(canonical_board)
        
        if step_count < 20:
            # 前 20 步忽略 5-8 层 (索引 4, 5, 6, 7)
            self._mask_layers(valid_moves, [4, 5, 6, 7])
        elif step_count < 60:
            # 中期忽略 7-8 层 (索引 6, 7)
            self._mask_layers(valid_moves, [6, 7])

        # 获取 MCTS 结果
        # 我们将更新后的 valid_moves 作为掩码传入，限制 MCTS 根节点的搜索方向
        probs = self.mcts.get_action_prob(canonical_board, temp=0, valid_mask=valid_moves)
        action = np.argmax(probs)
        
        # 恢复搜索量
        self.args.num_mcts_sims = original_sims
        
        # Convert simple integer action back to (layer, row, col)
        return self._action_to_coords(action)

    def _find_immediate_win_or_block(self, canonical_board):
        """检查是否有立即获胜或必须阻塞的位置"""
        valid_moves = self.game.get_valid_moves(canonical_board)
        
        # 1. 检查自己是否能立即获胜 (player = 1 in canonical)
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                next_board, _ = self.game.get_next_state(canonical_board, 1, a)
                if self.game.check_win(next_board, 1):
                    return a
        
        # 2. 检查对手是否能立即获胜并阻塞它 (opponent = -1 in canonical)
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                # 模拟对手下这一步
                next_board, _ = self.game.get_next_state(canonical_board, -1, a)
                if self.game.check_win(next_board, -1):
                    return a
        return None

    def _mask_layers(self, valid_moves, layers_to_ignore):
        """将指定层级的动作标记为无效"""
        for l in layers_to_ignore:
            start = l * 25
            end = (l + 1) * 25
            valid_moves[start:end] = 0

    def _action_to_coords(self, action):
        """将一维动作索引转换为 (layer, row, col)"""
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
