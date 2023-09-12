import pygame
# from Camera import Camera
from Automaton import *
import cv2, torch, numpy as np

# TODO : Migrate all this into a big class. That way the helper methods are easier to write.


# Initialize the pygame screen 
pygame.init()
el_size = 5
W,H = 200,200

font = pygame.font.SysFont(None, 25) 
# Load the images for the automaton
im_0 = pygame.transform.scale(pygame.image.load('graphics/star.png'), (el_size, el_size))



screen_W, screen_H = W*el_size, H*el_size

screen = pygame.display.set_mode((screen_W,screen_H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
# camera = Camera(W,H)

fps = 30

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.zeros((W,H,3),dtype=np.uint8)

# Initialize the automaton
auto = EmptyAuto((H,W))

updating = True
recording = False
launch_video = False

def draw_game_state(world_state,el_size):
    """
        Draws the game state on the screen, using the world_state array.
        For visualizing bigger pixels.

        Args:
            world_state (np.ndarray): (W,H,3) array of RGB uint8 values
            el_size (int): size of the pixels in the pygame window
        
        Returns:

    """
    use_graphics = False
    W,H,_ = world_state.shape
    surf = pygame.Surface((el_size*W, el_size*H))


    if(not use_graphics):
        for i in range(W):
            for j in range(H):
                color = world_state[i, j]  # Get the color from your array
                pygame.draw.rect(surf, color, (i*el_size, j*el_size, el_size, el_size))
    else :
        for i in range(W):
            for j in range(H):
                color = world_state[i, j]
                if(not (color==0).all()):
                    surf.blit(im_0, (i*el_size, j*el_size))

    #PIXEL VIEWING, EFFICIENT BUT UGLY. Need el_size=1.
    # surf = pygame.surfarray.make_surface(world_state)
    # screen.blit(surf, (0,0))

    return surf

while running:
    for event in pygame.event.get():
        # Event loop. Here we deal with all the interactivity
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_SPACE):
                updating=not updating
            if(event.key == pygame.K_r):
                recording= not recording
        # Handle the event loop for the camera (disabled for now)
        # camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        auto.step()
    
    #Retrieve the world_state from automaton
    world_state = auto.worldmap
    surface= draw_game_state(world_state,el_size)

    #For recording
    if(recording):
        if(not launch_video):
            launch_video = True
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            vid_loc = 'Videos/lgca1.mkv'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (W, H))

        frame_bgr = cv2.cvtColor(auto.worldmap, cv2.COLOR_RGB2BGR)
        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (W-10,H-10),2)
    
    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window (disabled for now)
    # zoomed_surface = camera.apply(surface)
    screen.blit(surface, (0, 0))
    clock.tick(fps)  # limits FPS
    curfps= clock.get_fps()
    fps_text = font.render("FPS: " + str(int(curfps)), True, (255, 0, 0))  # Red color

    screen.blit(fps_text, (10, 10))  # Display at position (10, 10)
    # Update the screen
    pygame.display.flip()




  

pygame.quit()
if(launch_video):
    video_out.release()