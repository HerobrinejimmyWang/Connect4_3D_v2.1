import pygame

# Colors
COLOR_BG = (11, 16, 38)
COLOR_PANEL = (28, 37, 65)
COLOR_PRIMARY = (0, 168, 232) 
COLOR_SECONDARY = (58, 80, 107)
COLOR_TEXT = (224, 224, 224)
COLOR_TEXT_DIM = (120, 130, 150)
COLOR_ACCENT = (0, 255, 153)
COLOR_WARNING = (255, 0, 85)
COLOR_HIGHLIGHT = (50, 200, 255)

class UIElement:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.active = True

    def draw(self, screen):
        pass
    
    def handle_event(self, event):
        pass

class Button(UIElement):
    def __init__(self, x, y, w, h, text, callback, font, bg_color=COLOR_PANEL, text_color=COLOR_TEXT, border=1):
        super().__init__(x, y, w, h)
        self.text = text
        self.callback = callback
        self.font = font
        self.bg_color = bg_color
        self.base_color = bg_color
        self.hover_color = tuple(min(c + 30, 255) for c in bg_color)
        self.text_color = text_color
        self.border_color = COLOR_PRIMARY
        self.border = border
        self.hovered = False
        
    def draw(self, screen):
        if not self.active:
            return # Hidden
            
        color = self.hover_color if self.hovered else self.bg_color
        
        # Glow effect if hovered
        if self.hovered:
            pygame.draw.rect(screen, COLOR_HIGHLIGHT, self.rect.inflate(4, 4), border_radius=4)
            
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        if self.border:
            pygame.draw.rect(screen, self.border_color, self.rect, self.border, border_radius=4)
        
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if not self.active:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.hovered:
                if self.callback:
                    self.callback()
                return True
        return False

class InputBox(UIElement):
    def __init__(self, x, y, w, h, font, placeholder="XYZ"):
        super().__init__(x, y, w, h)
        self.font = font
        self.text = ""
        self.placeholder = placeholder
        self.is_focused = False
        self.color_inactive = COLOR_SECONDARY
        self.color_active = COLOR_PRIMARY
        self.color = self.color_inactive
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_focused = not self.is_focused
            else:
                self.is_focused = False
            self.color = self.color_active if self.is_focused else self.color_inactive
            
        if event.type == pygame.KEYDOWN and self.is_focused:
            if event.key == pygame.K_RETURN:
                return self.text # Signal to process
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Accept numbers and spaces
                if len(self.text) < 10: # Limit length
                    self.text += event.unicode
        return None

    def draw(self, screen):
        # Background
        pygame.draw.rect(screen, tuple(max(0, c - 20) for c in COLOR_PANEL), self.rect)
        # Border
        pygame.draw.rect(screen, self.color, self.rect, 2)
        
        # Text
        display_text = self.text if self.text else self.placeholder
        text_col = COLOR_TEXT if self.text else COLOR_TEXT_DIM
        
        txt_surface = self.font.render(display_text, True, text_col)
        # Clip if too long?
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 10))

    def get_values(self):
        try:
            raw = self.text.strip()
            # If no spaces and 3 characters, split into individual digits
            if " " not in raw and len(raw) == 3:
                parts = list(raw)
            else:
                parts = raw.split()
            
            if len(parts) == 3:
                return [int(p) for p in parts]
        except ValueError:
            pass
        return None
    
    def clear(self):
        self.text = ""

class SelectionList(UIElement):
    def __init__(self, x, y, w, h, items, on_select_callback, font):
        super().__init__(x, y, w, h)
        self.items = items # list of dicts (needs 'name')
        self.callback = on_select_callback
        self.font = font
        self.visible = False
        self.scroll_y = 0
        self.item_height = 40
        
    def update_items(self, items):
        self.items = items
        
    def handle_event(self, event):
        if not self.visible: return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left click
                if self.rect.collidepoint(event.pos):
                    # Calculate index
                    rel_y = event.pos[1] - self.rect.y - self.scroll_y
                    idx = int(rel_y // self.item_height)
                    if 0 <= idx < len(self.items):
                        self.callback(idx)
                        self.visible = False
                    return True
                else:
                    # Click outside closes it
                    self.visible = False
                    # Don't consume event so other buttons work? 
                    # Usually modals consume the click so you don't accidentally click 'undo' behind it.
                    return True
            elif event.button == 4: # Scroll up
                self.scroll_y = min(0, self.scroll_y + 20)
                return True
            elif event.button == 5: # Scroll down
                max_content = len(self.items) * self.item_height
                if max_content > self.rect.height:
                    max_scroll = -(max_content - self.rect.height)
                    self.scroll_y = max(max_scroll, self.scroll_y - 20)
                return True
        return False

    def draw(self, screen):
        if not self.visible: return
        
        # Draw background shadow
        s = pygame.Surface((self.rect.width + 4, self.rect.height + 4))
        s.set_alpha(100)
        s.fill((0,0,0))
        screen.blit(s, (self.rect.x + 2, self.rect.y + 2))
        
        # Draw main BG
        pygame.draw.rect(screen, COLOR_PANEL, self.rect)
        pygame.draw.rect(screen, COLOR_PRIMARY, self.rect, 2)
        
        # Clip
        clip_rect = screen.get_clip()
        screen.set_clip(self.rect)
        
        mx, my = pygame.mouse.get_pos()
        
        for i, item in enumerate(self.items):
            y_pos = self.rect.y + self.scroll_y + i * self.item_height
            
            # Optimization: don't draw if out of view
            if y_pos + self.item_height < self.rect.y or y_pos > self.rect.bottom:
                continue
                
            item_rect = pygame.Rect(self.rect.x, y_pos, self.rect.width, self.item_height)
            
            # Hover
            if item_rect.collidepoint((mx, my)):
                pygame.draw.rect(screen, COLOR_SECONDARY, item_rect)
                
            text_name = item['name'] if isinstance(item, dict) else str(item)
            surf = self.font.render(text_name, True, COLOR_TEXT)
            # Centering vertically
            text_y = y_pos + (self.item_height - surf.get_height()) // 2
            screen.blit(surf, (self.rect.x + 10, text_y))
            
            # Divider
            pygame.draw.line(screen, COLOR_SECONDARY, (self.rect.x, y_pos+self.item_height-1), (self.rect.right, y_pos+self.item_height-1))
            
        screen.set_clip(clip_rect)

class ProgressBar(UIElement):
    def __init__(self, x, y, w, h, label_font):
        super().__init__(x, y, w, h)
        self.value = 0.5 # 0.0 to 1.0 (Win rate for Red)
        self.font = label_font
        self.visible = False
        
    def set_value(self, val):
        self.value = max(0.0, min(1.0, val))

    def draw(self, screen):
        if not self.visible:
            return
            
        # Background Bar
        pygame.draw.rect(screen, (50, 0, 0), self.rect, border_radius=4) # Dark Red base
        
        # Fill for Blue (Right side? Or Red Left?)
        # Let's say Value is Red Win Rate.
        # Draw Red portion
        red_w = int(self.rect.width * self.value)
        red_rect = pygame.Rect(self.rect.x, self.rect.y, red_w, self.rect.height)
        pygame.draw.rect(screen, (220, 60, 60), red_rect, border_top_left_radius=4, border_bottom_left_radius=4)
        
        # Draw Blue portion (Background is fine, but lets draw explicit blue for clarity)
        blue_rect = pygame.Rect(self.rect.x + red_w, self.rect.y, self.rect.width - red_w, self.rect.height)
        pygame.draw.rect(screen, (60, 150, 220), blue_rect, border_top_right_radius=4, border_bottom_right_radius=4)
        
        # Border
        pygame.draw.rect(screen, COLOR_TEXT_DIM, self.rect, 2, border_radius=4)
        
        # Text
        pct = int(self.value * 100)
        red_text = self.font.render(f"Red: {pct}%", True, COLOR_TEXT)
        blue_text = self.font.render(f"Blue: {100-pct}%", True, COLOR_TEXT)
        
        if self.rect.width > 200:
            screen.blit(red_text, (self.rect.x + 10, self.rect.y + 5))
            screen.blit(blue_text, (self.rect.right - blue_text.get_width() - 10, self.rect.y + 5))
