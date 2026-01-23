import pygame
import random
import tracking
import time

# Constant stuff
WIDTH, HEIGHT = 1024, 720
CELL_SIZE = 3
COLS, ROWS = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
SAND_COLOR = (0, 0, 0)
BG_COLOR = (0, 0, 0)
CURSOR_RADIUS = 40
SAND_PER_CLICK = 500
SCR = 10 # sand click radius
PHYSICS_ITERATIONS = 3

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sand Sim")
clock = pygame.time.Clock()
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

def update_sand():
    new_grid = [row[:] for row in grid]
    # Iterate from bottom to top to avoid multi-move in one frame
    for y in range(ROWS - 2, -1, -1):
        for x in range(COLS):
            if grid[y][x] != 0:  # Check if there's sand (not 0)
                # Try to fall straight down
                if grid[y+1][x] == 0:
                    new_grid[y][x] = 0
                    new_grid[y+1][x] = grid[y][x]
                # Try to fall diagonally left
                elif x > 0 and grid[y+1][x-1] == 0:
                    new_grid[y][x] = 0
                    new_grid[y+1][x-1] = grid[y][x]
                # Try to fall diagonally right
                elif x < COLS - 1 and grid[y+1][x+1] == 0:
                    new_grid[y][x] = 0
                    new_grid[y+1][x+1] = grid[y][x]
                # Otherwise, don't move (remove the sideways sliding)
    return new_grid


# game loop
running = True
while running:
    frame, handPos, posePos = tracking.generateVisuals()
    
    
    # Convert OpenCV BGR to RGB and transpose for pygame
    frame_rgb = frame[:, :, ::-1]  # BGR to RGB
    background_frame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(background_frame, (0, 0))
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # add sand on mouse click
    # if pygame.mouse.get_pressed()[0]:
        # mx, my = get_mouse_pos()
    if handPos is not None and len(handPos) > 8:
        mx, my = handPos[8][0] // CELL_SIZE, handPos[8][1] // CELL_SIZE
        sand_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        for _ in range(SAND_PER_CLICK):
            rx, ry = mx + random.randint(-SCR, SCR), my + random.randint(-SCR, SCR)
            if 0 <= rx < COLS and 0 <= ry < ROWS:
                grid[ry][rx] = sand_color

    grid = update_sand()
    for i in range(PHYSICS_ITERATIONS - 1):
        grid = update_sand()

    # Draw
    for y in range(ROWS):
        for x in range(COLS):
            if grid[y][x] != 0:
                pygame.draw.rect(screen, grid[y][x], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

