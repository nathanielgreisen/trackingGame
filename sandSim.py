import pygame
import random
import tracking
import io
import time

# Constant stuff
WIDTH, HEIGHT = 1024, 768
CELL_SIZE = 4
COLS, ROWS = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
SAND_COLOR = (0, 0, 0)
BG_COLOR = (0, 0, 0)
CURSOR_RADIUS = 40
SAND_PER_CLICK = 10
SCR = 3 # sand click radius

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sand Sim")
clock = pygame.time.Clock()
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

def get_mouse_pos():
    mx, my = pygame.mouse.get_pos()
    return mx // CELL_SIZE, my // CELL_SIZE

def update_sand():
    new_grid = [row[:] for row in grid]
    # Physics simulation logic here
    return new_grid

prev_time = time.time()

# game loop
running = True
while running:
    
    cv_frame = tracking.returnFrame(prev_time)
    # Convert OpenCV BGR to RGB and transpose for pygame
    frame_rgb = cv_frame[:, :, ::-1]  # BGR to RGB
    background_frame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(background_frame, (0, 0))
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # add sand on mouse click
    if pygame.mouse.get_pressed()[0]:
        mx, my = get_mouse_pos()
        for _ in range(SAND_PER_CLICK):
            rx, ry = mx + random.randint(-SCR, SCR), my + random.randint(-SCR, SCR)
            if 0 <= rx < COLS and 0 <= ry < ROWS:
                grid[ry][rx] = 1

    grid = update_sand()

    # Draw
    for y in range(ROWS):
        for x in range(COLS):
            if grid[y][x] == 1:
                pygame.draw.rect(screen, SAND_COLOR, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()