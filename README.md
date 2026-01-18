# Connect4 3D v2.1 (written by AI)

## Overview

Connect4 3D v2.1 is an enhanced version of the classic Connect Four game, extended into three dimensions with an integrated AlphaZero-style AI opponent. This project combines a fully interactive 3D/2D game interface with a self-play reinforcement learning system to train intelligent agents that can play 3D Connect Four at a competitive level.

### Key Features:
- **3D Gameplay**: Play Connect Four in a 5×5×8 grid (layers, rows, columns)
- **Dual View Modes**: Switch between 2D grid view and interactive 3D perspective view
- **AI Integration**: Built-in AlphaZero-style AI with Monte Carlo Tree Search (MCTS)
- **Multiple Game Modes**: Player vs Player, Player vs AI (AI plays as Red or Blue)
- **Training System**: Complete self-play training pipeline with parallel processing
- **Model Management**: Load, switch, and compare different AI generations
- **Visual Enhancements**: Win prediction bar, move hints, 3D rotation, winning line highlighting

## Project Structure

```
Connect4_3D_v2.1/
├── AI_* (folders)/          # AI training modules (multiple generations)
│   ├── game_rules.py       # Game logic and rules
│   ├── model.py            # Neural network architecture
│   ├── mcts.py             # Monte Carlo Tree Search
│   ├── checkpoints/        # Training checkpoints
│   └── save_model/         # Latest trained models
├── Connect4_AI_Enhanced.py # Main game application
├── game_ui_controls.py     # UI components (buttons, input boxes, etc.)
├── ai_interface.py         # Bridge between game and AI modules
├── main_train.py           # Training entry point
├── trainer.py              # Training logic with parallel processing
└── README.md               # This file
```

## Installation & Requirements

### Prerequisites:
- Python 3.7+
- Pygame
- PyTorch
- NumPy

### Install Dependencies:
```bash
pip install pygame torch numpy
```

## How to Run

### 1. Launch the Game:
```bash
python Connect4_AI_Enhanced.py
```

### 2. Start Training AI:
```bash
python main_train.py
```

**Note**: The AI training requires at least one `AI_*` folder in the project directory. If none exists, the game will run but AI features will be unavailable.

## Game Controls

### Menu:
- **Player vs Player**: Two human players take turns
- **PvAI (You Red)**: Human plays as Red, AI as Blue
- **PvAI (AI Red)**: AI plays as Red, Human as Blue
- **Model Selection**: Choose from different trained AI models

### In-Game:
- **2D View**: Click on cells to place pieces
- **3D View**: Drag to rotate, click coordinates or use input box
- **Toggle 3D View**: Switch between 2D and 3D perspectives
- **Get Hint**: Shows AI's suggested move (yellow flash)
- **Undo (U)**: Undo last move(s) (2 moves in PvAI mode)
- **Restart (R)**: Reset the game
- **Win Prediction**: Toggle win probability display
- **Back to Menu**: Return to main menu

### Coordinate Input (3D View):
Enter three numbers (L R C) in the input box, e.g., "3 2 4" for Layer 3, Row 2, Column 4.

## AI Training System

The project includes a complete AlphaZero-style training pipeline:

### Training Process:
1. **Self-Play**: Multiple parallel games generate training data
2. **Neural Network Training**: 3D convolutional network learns from games
3. **Evaluation**: Model plays against random agents to measure progress
4. **Checkpointing**: Models saved at regular intervals

### Customize Training:
Edit `main_train.py` to adjust:
- `num_iterations`: Total training cycles
- `num_self_play_games`: Games per iteration
- `checkpoint_interval`: Save frequency
- `learning_rate`: Training speed

## 3D Visualization

The 3D view features:
- Interactive rotation (click and drag)
- Color-coded axes (L=Green, R=Blue, C=Red)
- Wireframe bounding box
- Depth-sorted cube rendering
- Real-time piece placement visualization

## Performance Notes

- **AI Response Time**: MCTS simulations (~100) provide quick but thoughtful moves
- **Training Speed**: Parallel processing utilizes 50% of CPU cores
- **Memory Usage**: Moderate - depends on network size and batch settings
- **GPU Support**: Optional - PyTorch will use CUDA if available

## Customization

### Modify Game Rules:
Edit `game_rules.py` in any AI folder to change:
- Board dimensions
- Win conditions
- Move validation

### Change AI Architecture:
Edit `model.py` to adjust:
- Neural network depth
- Convolutional filters
- Residual blocks

### Adjust UI:
Edit `game_ui_controls.py` for:
- Color schemes
- Button styles
- Layout parameters

## Troubleshooting

### Common Issues:

1. **No AI folders found**:
   - Ensure at least one `AI_*` folder exists
   - Copy the provided AI template if missing

2. **Model fails to load**:
   - Check PyTorch version compatibility
   - Verify model checkpoint integrity

3. **3D view performance issues**:
   - Reduce FPS in settings
   - Simplify 3D rendering in code

4. **Training stalls**:
   - Reduce `num_self_play_games`
   - Check available system memory



# Connect4 3D v2.1 中文文档（AI撰写）

## 项目概述

Connect4 3D v2.1 是经典四子棋的增强版本，扩展到三维空间并集成了类AlphaZero风格的AI对手。该项目结合了完全交互式的3D/2D游戏界面与自我对弈的强化学习系统，能够训练出具有竞争力的3D四子棋智能体。

### 主要特性：
- **3D游戏玩法**：在5×5×8网格（层、行、列）中进行四子棋对战
- **双重视图模式**：在2D网格视图和交互式3D透视视图之间切换
- **AI集成**：内置类AlphaZero风格AI，使用蒙特卡洛树搜索（MCTS）
- **多种游戏模式**：玩家对战、玩家对AI（AI可执红或执蓝）
- **训练系统**：完整的自我对弈训练流程，支持并行处理
- **模型管理**：加载、切换和比较不同的AI版本
- **视觉增强**：胜率预测条、移动提示、3D旋转、胜利连线高亮

## 项目结构

```
Connect4_3D_v2.1/
├── AI_* (文件夹)/          # AI训练模块（多代版本）
│   ├── game_rules.py       # 游戏逻辑和规则
│   ├── model.py            # 神经网络架构
│   ├── mcts.py             # 蒙特卡洛树搜索
│   ├── checkpoints/        # 训练检查点
│   └── save_model/         # 最新训练模型
├── Connect4_AI_Enhanced.py # 主游戏应用程序
├── game_ui_controls.py     # UI组件（按钮、输入框等）
├── ai_interface.py         # 游戏与AI模块的桥梁
├── main_train.py           # 训练入口点
├── trainer.py              # 并行处理训练逻辑
└── README.md               # 本文档
```

## 安装与要求

### 先决条件：
- Python 3.7+
- Pygame
- PyTorch
- NumPy

### 安装依赖：
```bash
pip install pygame torch numpy
```

## 运行方法

### 1. 启动游戏：
```bash
python Connect4_AI_Enhanced.py
```

### 2. 开始训练AI：
```bash
python main_train.py
```

**注意**：AI训练需要项目目录中至少有一个`AI_*`文件夹。如果不存在，游戏可以运行但AI功能将不可用。

## 游戏控制

### 主菜单：
- **玩家对战**：两名人类玩家轮流下棋
- **人机对战（你执红）**：人类执红，AI执蓝
- **人机对战（AI执红）**：AI执红，人类执蓝
- **模型选择**：从不同的训练AI模型中选择

### 游戏中：
- **2D视图**：点击格子放置棋子
- **3D视图**：拖拽旋转，点击坐标或使用输入框
- **切换3D视图**：在2D和3D视角之间切换
- **获取提示**：显示AI建议的走法（黄色闪烁）
- **撤销（U）**：撤销最后一步（人机模式下撤销两步）
- **重新开始（R）**：重置游戏
- **胜率预测**：切换胜率概率显示
- **返回菜单**：返回主菜单

### 坐标输入（3D视图）：
在输入框中输入三个数字（L R C），例如"3 2 4"表示第3层、第2行、第4列。

## AI训练系统

项目包含完整的类AlphaZero风格训练流程：

### 训练流程：
1. **自我对弈**：多个并行游戏生成训练数据
2. **神经网络训练**：3D卷积网络从游戏中学习
3. **评估**：模型与随机智能体对弈以衡量进度
4. **检查点保存**：定期保存模型

### 自定义训练：
编辑`main_train.py`调整：
- `num_iterations`：总训练周期数
- `num_self_play_games`：每轮游戏数
- `checkpoint_interval`：保存频率
- `learning_rate`：训练速度

## 3D可视化

3D视图特性：
- 交互式旋转（点击拖拽）
- 颜色编码的坐标轴（L=绿，R=蓝，C=红）
- 线框边界框
- 深度排序的立方体渲染
- 实时棋子放置可视化

## 性能说明

- **AI响应时间**：MCTS模拟（约100次）提供快速但深思熟虑的走法
- **训练速度**：并行处理利用50%的CPU核心
- **内存使用**：中等 - 取决于网络大小和批次设置
- **GPU支持**：可选 - PyTorch在可用时会使用CUDA

## 自定义

### 修改游戏规则：
编辑任意AI文件夹中的`game_rules.py`来更改：
- 棋盘尺寸
- 胜利条件
- 移动验证

### 更改AI架构：
编辑`model.py`调整：
- 神经网络深度
- 卷积滤波器
- 残差块

### 调整UI：
编辑`game_ui_controls.py`修改：
- 配色方案
- 按钮样式
- 布局参数

## 故障排除

### 常见问题：

1. **找不到AI文件夹**：
   - 确保至少有一个`AI_*`文件夹存在
   - 如果缺失，请复制提供的AI模板

2. **模型加载失败**：
   - 检查PyTorch版本兼容性
   - 验证模型检查点完整性

3. **3D视图性能问题**：
   - 降低设置中的FPS
   - 简化代码中的3D渲染

4. **训练停滞**：
   - 减少`num_self_play_games`
   - 检查可用系统内存