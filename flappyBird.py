import pygame
import sys
import random
import tracking

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 0.5
FLAP_STRENGTH = -12
PIPE_WIDTH = 80
PIPE_GAP = 150
PIPE_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GREEN = (34, 139, 34)

class Bird:
    def __init__(self):
        self.width = 30
        self.height = 30
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0

    def update(self, finger_x, finger_y):
        # Move bird toward finger position
        target_x = finger_x if finger_x else self.x
        target_y = finger_y if finger_y else self.y
        
        self.x += (target_x - self.x) * 0.1  # Smooth movement
        self.y += (target_y - self.y) * 0.1

    def draw(self, screen):
        pygame.draw.rect(screen, BLUE, (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height))

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap_y = random.randint(PIPE_GAP + 50, SCREEN_HEIGHT - PIPE_GAP - 50)
        self.scored = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen):
        # Top pipe
        pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.gap_y - PIPE_GAP // 2))
        # Bottom pipe
        pygame.draw.rect(screen, GREEN, (self.x, self.gap_y + PIPE_GAP // 2, self.width, SCREEN_HEIGHT - self.gap_y - PIPE_GAP // 2))

    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, self.width, SCREEN_HEIGHT - self.gap_y - PIPE_GAP // 2)
        return top_rect, bottom_rect

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Finger Tracking")
    clock = pygame.time.Clock()
    
    bird = Bird()
    pipes = []
    score = 0
    game_over = False
    pipe_counter = 0

    font = pygame.font.Font(None, 36)

    while True:
        
        frame, handPos, posePos = tracking.generateVisuals()
        
        # Convert OpenCV BGR to RGB and transpose for pygame
        frame_rgb = frame[:, :, ::-1]  # BGR to RGB
        background_frame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(background_frame, (0, 0))
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_over:
                game_over = False
                bird = Bird()
                pipes = []
                score = 0

        if not game_over:
            # Get finger position from tracking.py
            finger_x, finger_y = None, None
            if handPos is not None and len(handPos) > 8:
                finger_x, finger_y = handPos[8][0], handPos[8][1]
            
            bird.update(finger_x, finger_y)

            pipe_counter += 1
            if pipe_counter > 100:
                pipes.append(Pipe(SCREEN_WIDTH))
                pipe_counter = 0

            for pipe in pipes[:]:
                pipe.update()
                if pipe.x + pipe.width < 0:
                    pipes.remove(pipe)
                
                if not pipe.scored and pipe.x < bird.x:
                    score += 1
                    pipe.scored = True

                # Collision detection
                top_rect, bottom_rect = pipe.get_rects()
                if bird.get_rect().colliderect(top_rect) or bird.get_rect().colliderect(bottom_rect):
                    game_over = True

            # Check boundaries
            if bird.y < 0 or bird.y > SCREEN_HEIGHT:
                game_over = True

        # Draw
        # screen.fill(WHITE)
        
        for pipe in pipes:
            pipe.draw(screen)
        
        bird.draw(screen)
        
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))
        
        if game_over:
            game_over_text = font.render("Game Over! Press SPACE to restart", True, BLACK)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()