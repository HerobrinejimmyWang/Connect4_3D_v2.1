import pygame
import sys
import numpy as np
import math
import os

# Custom modules
from game_ui_controls import *
from ai_interface import AI_Interface

# --- Configuration ---
WINDOW_WIDTH = 1350
WINDOW_HEIGHT = 800
FPS = 60
BOARD_SIZE = 5
MAX_LAYERS = 8

# Colors
COLOR_P1 = (220, 60, 60)   # Red
COLOR_P2 = (60, 150, 220)  # Blue
COLOR_GRID = (50, 60, 80)
COLOR_HINT = (255, 255, 0) # Yellow

class GameManager:
    def __init__(self):
        self.ai_interface = AI_Interface()
        self.reset_game()
        
        # Discover models
        self.available_models = self.ai_interface.find_checkpoints()
        if self.available_models:
            # Load best model by default
            self.current_model_idx = 0
            self.load_model_by_idx(0)
        else:
            self.current_model_idx = -1
            
        # Modes: 'PVP', 'PVE_HUMAN_RED', 'PVE_AI_RED'
        self.game_mode = 'PVP' 
        
    def reset_game(self):
        self.board = np.zeros((MAX_LAYERS, BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1 # 1 or 2
        self.history = [] # Stack of (layer, row, col, player)
        self.game_over = False
        self.winner = 0
        self.winning_line = []
        self.prediction_enabled = False
        self.prediction_value = 0.5
        self.hint_pos = None

    def load_model_by_idx(self, idx):
        if 0 <= idx < len(self.available_models):
            path = self.available_models[idx]['path']
            success = self.ai_interface.load_model(path)
            if success:
                self.current_model_idx = idx
                print(f"Loaded: {self.available_models[idx]['name']}")
                return True
        return False
        
    def is_valid_move(self, layer, row, col):
        if not (0 <= layer < MAX_LAYERS and 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[layer, row, col] != 0:
            return False
        # Gravity
        if layer == 0: return True
        return self.board[layer-1, row, col] != 0

    def make_move(self, layer, row, col):
        if self.game_over or not self.is_valid_move(layer, row, col):
            return False
        
        self.board[layer, row, col] = self.current_player
        self.history.append((layer, row, col, self.current_player))
        
        # Win check logic (Simplified from v1.4 logic for brevity but robust)
        # Using AI Interface GameRules to double check (or reimplement fast here)
        # Re-using the logic from the provided snippets is safer.
        if self.check_win_at(layer, row, col):
            self.game_over = True
            self.winner = self.current_player
        else:
            self.current_player = 3 - self.current_player
        
        # Reset Hint
        self.hint_pos = None
        
        # Update Prediction if enabled
        if self.prediction_enabled and not self.game_over:
            self.update_prediction()
           
        return True

    def undo(self):
        if not self.history: return False
        
        # How many steps to undo?
        steps = 1
        if "PVE" in self.game_mode:
            # If PVE, usually we undo 2 steps (Human + AI) so it's Human's turn again
            # Unless game ended on Human turn?
            # Or if it's currently AI turn (unlikely as AI moves instantly usually), undo 1
            # Simple logic: undo until it is Human's turn?
            # Or just Undo 1 step is strict. Prompt says:
            # "If vs AI, undo to Human's last step" -> Means undo AI move AND Human move if AI moved.
            
            # Check who made the last move
            last_mover = self.history[-1][3]
            ai_role = 1 if 'AI_RED' in self.game_mode else 2
            
            if last_mover == ai_role:
                steps = 2 # Undo AI then Human
            else:
                steps = 1 # Just Human moved (maybe AI is thinking or game start)
                
        # Perform undo
        for _ in range(steps):
            if not self.history: break
            l, r, c, p = self.history.pop()
            self.board[l, r, c] = 0
        
        self.game_over = False
        self.winner = 0
        self.winning_line = []
        
        # Reset player
        if self.history:
            self.current_player = 3 - self.history[-1][3]
        else:
             # Reset to initial player based on mode
             if self.game_mode == 'PVE_AI_RED':
                 self.current_player = 1 # AI starts, but we need to trigger AI loop?
                 # Actually if we undid all, AI needs to move. 
                 # Handled in main loop
             else:
                 self.current_player = 1

        if self.prediction_enabled:
            self.update_prediction()
            
        return True

    def update_prediction(self):
        # Run in thread? For now blocking is okay as inference is fast on small net
        self.prediction_value = self.ai_interface.get_prediction(self.board, self.current_player)

    def get_hint(self):
        if self.game_over: return
        move = self.ai_interface.get_best_move(self.board, self.current_player)
        if move:
            self.hint_pos = move

    def check_win_at(self, layer, row, col):
        # Direction vectors (dz, dy, dx)
        directions = [
            (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        player = self.board[layer, row, col]
        
        for dz, dy, dx in directions:
            # Check line
            count = 1
            line = [(layer, row, col)]
            
            # Forward
            for i in range(1, 4):
                z, y, x = layer + i*dz, row + i*dy, col + i*dx
                if 0<=z<MAX_LAYERS and 0<=y<BOARD_SIZE and 0<=x<BOARD_SIZE and self.board[z,y,x] == player:
                    count += 1
                    line.append((z, y, x))
                else: break
            
            # Backward
            for i in range(1, 4):
                z, y, x = layer - i*dz, row - i*dy, col - i*dx
                if 0<=z<MAX_LAYERS and 0<=y<BOARD_SIZE and 0<=x<BOARD_SIZE and self.board[z,y,x] == player:
                    count += 1
                    line.append((z, y, x))
                else: break
            
            if count >= 4:
                self.winning_line = line
                return True
        return False

class Connect43DApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Connect 4 3D - AI Enhanced")
        self.clock = pygame.time.Clock()
        self.font_curr = pygame.font.SysFont("Consolas", 20)
        self.font_title = pygame.font.SysFont("Impact", 40)
        
        self.gm = GameManager()
        self.state = "MENU" # MENU, GAME
        
        # UI Elements
        self.setup_ui()
        
        # 3D View vars
        self.view_3d_active = False
        self.rot_x = 30 # Pitch
        self.rot_y = 30 # Yaw
        self.mouse_drag_active = False
        self.last_mouse_pos = (0,0)
        
        # Flashing
        self.flash_timer = 0
        
    def setup_ui(self):
        cx = WINDOW_WIDTH // 2
        cy = WINDOW_HEIGHT // 2
        
        # --- MENU UI ---
        self.btn_pvp = Button(cx - 100, cy - 60, 200, 40, "Player vs Player", lambda: self.start_game('PVP'), self.font_curr)
        self.btn_pve_h = Button(cx - 100, cy, 200, 40, "PvAI (You Red)", lambda: self.start_game('PVE_HUMAN_RED'), self.font_curr)
        self.btn_pve_a = Button(cx - 100, cy + 60, 200, 40, "PvAI (AI Red)", lambda: self.start_game('PVE_AI_RED'), self.font_curr)
        
        self.btn_model = Button(cx - 150, cy + 120, 300, 40, "Model: Default", self.show_model_list, self.font_curr)
        
        # Model List (Initially Hidden)
        items = self.gm.available_models
        self.list_models = SelectionList(cx - 150, cy + 160, 300, 200, items, self.on_model_selected, self.font_curr)
        
        self.update_model_btn_text()
        
        # --- GAME UI ---
        # Right sidebar controls
        rx = WINDOW_WIDTH - 250
        
        self.btn_view = Button(rx, 50, 200, 40, "Toggle 3D View", self.toggle_view, self.font_curr)
        self.btn_hint = Button(rx, 110, 200, 40, "Get Hint", self.gm.get_hint, self.font_curr)
        self.btn_undo = Button(rx, 170, 200, 40, "Undo (U)", self.gm.undo, self.font_curr)
        self.btn_restart = Button(rx, 230, 200, 40, "Restart (R)", self.gm.reset_game, self.font_curr)
        self.btn_pred = Button(rx, 290, 200, 40, "Win Prediction", self.toggle_prediction, self.font_curr)
        self.btn_menu = Button(rx, WINDOW_HEIGHT - 60, 200, 40, "Back to Menu", self.to_menu, self.font_curr, bg_color=(80, 20, 20))
        
        self.input_coords = InputBox(rx, 400, 140, 40, self.font_curr, "L R C")
        self.btn_input_go = Button(rx + 150, 400, 50, 40, "GO", self.process_coord_input, self.font_curr)
        
        self.progress_win = ProgressBar(WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT - 100, 400, 30, self.font_curr)

    def update_model_btn_text(self):
        if self.gm.current_model_idx >= 0:
            name = self.gm.available_models[self.gm.current_model_idx]['name']
            self.btn_model.text = f"Model: {name}"
        else:
            self.btn_model.text = "Model: None Found"

    def show_model_list(self):
        self.list_models.visible = not self.list_models.visible

    def on_model_selected(self, idx):
        if self.gm.load_model_by_idx(idx):
            self.update_model_btn_text()

    def start_game(self, mode):
        self.gm.game_mode = mode
        self.gm.reset_game()
        self.state = "GAME"
        # If AI is first
        if mode == 'PVE_AI_RED':
            self.gm.current_player = 1
            # Trigger AI move immediately? 
            # We do it in update loop

    def to_menu(self):
        self.state = "MENU"

    def toggle_view(self):
        self.view_3d_active = not self.view_3d_active

    def toggle_prediction(self):
        self.gm.prediction_enabled = not self.gm.prediction_enabled
        if self.gm.prediction_enabled:
            self.gm.update_prediction()

    def process_coord_input(self):
        vals = self.input_coords.get_values()
        if vals:
            # User input: 3 numbers. Prompt says "Length Width Height" -> "5x5x8".
            # Mapping: vals[0] -> X (Col), vals[1] -> Y (Row), vals[2] -> Z (Layer)
            # Count from 1.
            # Let's assume input order is: Layer Row Col as displayed in HUD?
            # Or "l r c" placeholder suggests Layer Row Col.
            l, r, c = vals[0]-1, vals[1]-1, vals[2]-1
            
            if self.gm.make_move(l, r, c):
                self.input_coords.clear()
            else:
                self.input_coords.clear() # Invalid
                print("Invalid Move via Coords")

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                
            if self.state == "MENU":
                # If list open, handle it first
                if self.list_models.visible:
                    if self.list_models.handle_event(event):
                         return # Consumed
                
                for btn in [self.btn_pvp, self.btn_pve_h, self.btn_pve_a, self.btn_model]:
                    btn.handle_event(event)
                    
            elif self.state == "GAME":
                # Global Keys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: self.gm.reset_game()
                    elif event.key == pygame.K_u: self.gm.undo()
                
                # UI Buttons
                for btn in [self.btn_view, self.btn_hint, self.btn_undo, self.btn_restart, self.btn_pred, self.btn_menu, self.btn_input_go]:
                    btn.handle_event(event)
                
                # Input Box
                res = self.input_coords.handle_event(event)
                if res: self.process_coord_input()
                
                # 3D Interaction
                if self.view_3d_active:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Check if over board area (Left side)
                        if event.pos[0] < WINDOW_WIDTH - 250:
                            self.mouse_drag_active = True
                            self.last_mouse_pos = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.mouse_drag_active = False
                    elif event.type == pygame.MOUSEMOTION and self.mouse_drag_active:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.rot_y += dx * 0.5
                        self.rot_x += dy * 0.5
                        self.last_mouse_pos = event.pos
                else:
                    # 2D Interaction - Click to place
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # Only if not clicking UI
                        if event.pos[0] < WINDOW_WIDTH - 250:
                            self.handle_2d_click(event.pos)

    def handle_2d_click(self, pos):
        if self.gm.game_over: return
        
        # Determine cell from pos
        # Layout: 4 cols of 2 layers
        CELL = 40
        GAP = 5
        GRP_GAP = 20
        MARGIN = 40
        
        mx, my = pos
        
        # It's tricky to map exactly without re-calculating the layout logic.
        layers_per_row = 4
        
        # Grid block width
        grid_w = BOARD_SIZE * (CELL + GAP)
        block_w = grid_w + GRP_GAP
        
        for l in range(MAX_LAYERS):
            row_idx = l // layers_per_row
            col_idx = l % layers_per_row
            
            start_x = MARGIN + col_idx * block_w
            start_y = MARGIN + row_idx * (block_w + 30) # Height is roughly width
            
            if start_x <= mx < start_x + grid_w and start_y <= my < start_y + grid_w:
                # Inside a grid
                c = (mx - start_x) // (CELL + GAP)
                r = (my - start_y) // (CELL + GAP)

                # Ensure r, c are valid ints
                r = int(r)
                c = int(c)
                
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    self.gm.make_move(l, r, c)
                return

    def update(self):
        self.flash_timer += 1
        
        # AI Turn Logic
        if self.state == "GAME" and not self.gm.game_over:
            is_ai_turn = False
            if self.gm.game_mode == 'PVE_HUMAN_RED':
                if self.gm.current_player == 2: is_ai_turn = True
            elif self.gm.game_mode == 'PVE_AI_RED':
                if self.gm.current_player == 1: is_ai_turn = True
                
            if is_ai_turn:
                # Add delay or thread this so UI doesn't freeze
                # For simplicity, 1 frame blocking
                # To prevent instant moves, check timer or something
                move = self.gm.ai_interface.get_best_move(self.gm.board, self.gm.current_player)
                if move:
                    l, r, c = move
                    self.gm.make_move(l, r, c)

    def draw(self):
        self.screen.fill(COLOR_BG)
        
        if self.state == "MENU":
            self.draw_menu()
        else:
            self.draw_game()
        
        pygame.display.flip()

    def draw_menu(self):
        # Title
        t_surf = self.font_title.render("CONNECT 4 3D | AI EDITION", True, COLOR_PRIMARY)
        self.screen.blit(t_surf, (WINDOW_WIDTH//2 - t_surf.get_width()//2, 100))
        
        for btn in [self.btn_pvp, self.btn_pve_h, self.btn_pve_a, self.btn_model]:
            btn.draw(self.screen)
        
        # Draw list on top if visible
        self.list_models.draw(self.screen)

    def draw_game(self):
        # Draw UI Panel background
        pygame.draw.rect(self.screen, COLOR_PANEL, (WINDOW_WIDTH - 250, 0, 250, WINDOW_HEIGHT))
        
        # Draw Board
        if self.view_3d_active:
            self.draw_board_3d()
        else:
            self.draw_board_2d()
            
        # Draw UI Controls
        for btn in [self.btn_view, self.btn_hint, self.btn_undo, self.btn_restart, self.btn_pred, self.btn_menu]:
            btn.draw(self.screen)
            
        if self.view_3d_active:
            # Draw input box/label
            lbl = self.font_curr.render("Coord Input (L R C):", True, COLOR_TEXT)
            self.screen.blit(lbl, (self.input_coords.rect.x, self.input_coords.rect.y - 25))
            self.input_coords.draw(self.screen)
            self.btn_input_go.draw(self.screen)
            
        # Draw Prediction Bar
        if self.gm.prediction_enabled:
            # Update value
            self.progress_win.visible = True
            self.progress_win.set_value(self.gm.prediction_value)
            self.progress_win.draw(self.screen)
        
        # Draw Info Text
        p_text = f"Turn: {'RED' if self.gm.current_player == 1 else 'BLUE'}"
        p_col = COLOR_P1 if self.gm.current_player == 1 else COLOR_P2
        if self.gm.game_over:
            p_text = f"Winner: {'RED' if self.gm.winner == 1 else 'BLUE'}"
            if self.gm.winner == 0: p_text = "DRAW"
        
        info_surf = self.font_title.render(p_text, True, p_col)
        self.screen.blit(info_surf, (20, WINDOW_HEIGHT - 60))

    def draw_board_2d(self):
        # Same logic as original but cleaner
        CELL = 40
        GAP = 5
        GRP_GAP = 20
        MARGIN = 40
        layers_per_row = 4
        block_w = BOARD_SIZE * (CELL + GAP) + GRP_GAP
        
        for l in range(MAX_LAYERS):
            row_idx = l // layers_per_row
            col_idx = l % layers_per_row
            sx = MARGIN + col_idx * block_w
            sy = MARGIN + row_idx * (block_w + 30)
            
            # Label
            lbl = self.font_curr.render(f"Layer {l+1}", True, COLOR_TEXT_DIM)
            self.screen.blit(lbl, (sx, sy - 25))
            
            # Grid
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    x = sx + c * (CELL + GAP)
                    y = sy + r * (CELL + GAP)
                    
                    val = self.gm.board[l, r, c]
                    col = COLOR_GRID
                    if val == 1: col = COLOR_P1
                    elif val == 2: col = COLOR_P2
                    
                    # Highlight winning
                    if (l,r,c) in self.gm.winning_line:
                        pygame.draw.rect(self.screen, (255, 215, 0), (x-2, y-2, CELL+4, CELL+4), 3)
                        
                    # Hint
                    if self.gm.hint_pos == (l,r,c) and (self.flash_timer % 30 < 15):
                        pygame.draw.rect(self.screen, COLOR_HINT, (x, y, CELL, CELL), 3)

                    pygame.draw.rect(self.screen, col, (x, y, CELL, CELL), border_radius=4)


    def draw_board_3d(self):
        # 3D Projection Logic
        # Center of viewport
        cx, cy = (WINDOW_WIDTH - 250) // 2, WINDOW_HEIGHT // 2
        scale = 30 # Size of cubes
        
        # Gather all filled cells and hint/ghosts
        cubes = [] # (depth, draw_func)
        
        # Center point offsets (to rotate around center)
        # Using centered coordinates for rotation
        # Box center is at (3.5, 2.0, 2.0) for 8x5x5
        b_l, b_r, b_c = 8.0, 5.0, 5.0
        off_l = (b_l - 1) / 2.0
        off_r = (b_r - 1) / 2.0
        off_c = (b_c - 1) / 2.0
        
        # Rotations
        rad_x = math.radians(self.rot_x)
        rad_y = math.radians(self.rot_y)
        cos_x, sin_x = math.cos(rad_x), math.sin(rad_x)
        cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
        
        def project_point(l, r, c):
            # l->y (vertical), r->z (depth), c->x (horizontal)
            mx = (c - off_c) * 1.5
            my = -(l - off_l) * 1.5 # Invert Y for screen coords
            mz = (r - off_r) * 1.5
            
            # Rotate Y (Yaw)
            x1 = mx * cos_y - mz * sin_y
            z1 = mx * sin_y + mz * cos_y
            y1 = my
            
            # Rotate X (Pitch)
            y2 = y1 * cos_x - z1 * sin_x
            z2 = y1 * sin_x + z1 * cos_x
            x2 = x1
            
            # Screen
            sx = cx + x2 * scale
            sy = cy + y2 * scale
            return sx, sy, z2

        # 1. Draw Axis and Box Wireframe
        # Box corners: (0,0,0) to (7,4,4)
        # But we want to indicate the full volume capacity: 0..8, 0..5, 0..5
        # The indices are 0..BOARD_SIZE-1.
        # Let's draw the bounding box of cell centers or cell volumes?
        # Cell centers go from 0 to MAX.
        
        # Axis Lines: From Origin (-1,-1,-1 relative to first cell) to visualize direction?
        # Let's imply the coordinate system is based on the grid indices.
        # Origin is logic (0,0,0).
        
        origin_pt = project_point(-0.5, -0.5, -0.5)
        
        # Draw Axes
        # L Axis (Vertical): Green
        l_end = project_point(8.0, -0.5, -0.5)
        pygame.draw.line(self.screen, (0, 255, 0), (origin_pt[0], origin_pt[1]), (l_end[0], l_end[1]), 2)
        
        # R Axis (Depth): Blue
        r_end = project_point(-0.5, 5.0, -0.5)
        pygame.draw.line(self.screen, (100, 100, 255), (origin_pt[0], origin_pt[1]), (r_end[0], r_end[1]), 2)
        
        # C Axis (Horizontal): Red
        c_end = project_point(-0.5, -0.5, 5.0)
        pygame.draw.line(self.screen, (255, 100, 100), (origin_pt[0], origin_pt[1]), (c_end[0], c_end[1]), 2)
        
        # Draw Labels
        font = self.font_curr
        self.screen.blit(font.render("L", True, (0, 255, 0)), (l_end[0], l_end[1]))
        self.screen.blit(font.render("R", True, (100, 100, 255)), (r_end[0], r_end[1]))
        self.screen.blit(font.render("C", True, (255, 100, 100)), (c_end[0], c_end[1]))
        
        # Draw Wireframe Box (lightly)
        corners = [
            (-0.5, -0.5, -0.5), (7.5, -0.5, -0.5), (7.5, 4.5, -0.5), (-0.5, 4.5, -0.5), # Back Face
            (-0.5, -0.5, 4.5), (7.5, -0.5, 4.5), (7.5, 4.5, 4.5), (-0.5, 4.5, 4.5)  # Front Face
        ]
        proj_corners = [project_point(*p) for p in corners]
        
        # Draw edges connecting corners
        box_col = (50, 60, 80)
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Back
            (4,5), (5,6), (6,7), (7,4), # Front
            (0,4), (1,5), (2,6), (3,7)  # Connecting
        ]
        for s, e in edges:
            p1 = proj_corners[s]
            p2 = proj_corners[e]
            pygame.draw.line(self.screen, box_col, (p1[0], p1[1]), (p2[0], p2[1]), 1)

        # 2. Draw Cubes
        for l in range(MAX_LAYERS):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    val = self.gm.board[l, r, c]
                    is_hint = (self.gm.hint_pos == (l,r,c))
                    
                    if val != 0 or is_hint:
                        px, py, depth = project_point(l, r, c)
                        
                        col = COLOR_GRID
                        if val == 1: col = COLOR_P1
                        elif val == 2: col = COLOR_P2
                        
                        if is_hint:
                            if self.flash_timer % 30 < 15:
                                col = COLOR_HINT
                            else: continue # Don't draw
                        
                        # Store for sorting
                        cubes.append({'z': depth, 'pos': (px, py), 'color': col, 'val': val})
        
        # Sort by depth (farthest first). Large positive Z is close or far? 
        # Z goes into screen in standard RHS if Y is up... 
        # Actually just sort descending Z usually works if Z is "into screen". 
        # If Z is "towards viewer", sort ascending.
        # Let's try sorting descending z first.
        cubes.sort(key=lambda k: k['z'], reverse=False) 
        
        # Draw
        for cube in cubes:
            x, y = cube['pos']
            s = scale // 1.2 # slightly smaller than grid
            pygame.draw.rect(self.screen, cube['color'], (x - s//2, y - s//2, s, s))
            # Add simple shading/outline
            pygame.draw.rect(self.screen, (255,255,255), (x - s//2, y - s//2, s, s), 1)
            
            # Letter for layer?
            # if cube['val'] != 0:
            #     pygame.draw.circle(self.screen, (0,0,0), (x,y), 2)
        
        if self.view_3d_active:
             # Draw axis labels explaining keys near the input box?
             pass

if __name__ == "__main__":
    app = Connect43DApp()
    app.run()
