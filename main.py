import pygame
from Camera import Camera
from Automaton import *
import cv2, torch, numpy as np

# TODO : Migrate all this into a big class. That way the helper methods are easier to write.


# Initialize the pygame screen 
pygame.init()
el_size = 2
W,H = 100,100

font = pygame.font.SysFont(None, 25) 
graph_folder = 'new_graph/'
# Load the images for the automaton
xm = pygame.transform.scale(pygame.image.load(graph_folder+'xm.png'), (el_size, el_size))
xp = pygame.transform.scale(pygame.image.load(graph_folder+'xp.png'), (el_size, el_size))
ym = pygame.transform.scale(pygame.image.load(graph_folder+'ym.png'), (el_size, el_size))
yp = pygame.transform.scale(pygame.image.load(graph_folder+'yp.png'), (el_size, el_size))

sxm = pygame.transform.scale(pygame.image.load(graph_folder+'sxm.png'), (el_size, el_size))
sxp = pygame.transform.scale(pygame.image.load(graph_folder+'sxp.png'), (el_size, el_size))
sym = pygame.transform.scale(pygame.image.load(graph_folder+'sym.png'), (el_size, el_size))
syp = pygame.transform.scale(pygame.image.load(graph_folder+'syp.png'), (el_size, el_size))

conf = pygame.transform.scale(pygame.image.load(graph_folder+'conf0.png'), (el_size, el_size))
sens0 = pygame.transform.scale(pygame.image.load(graph_folder+'sens0.png'), (el_size, el_size))
excited = pygame.transform.scale(pygame.image.load(graph_folder+'excited.png'), (el_size, el_size))
conf01 = pygame.transform.scale(pygame.image.load(graph_folder+'conf01.png'), (el_size, el_size))

screen_W, screen_H = W*el_size, H*el_size

screen = pygame.display.set_mode((screen_W,screen_H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(screen_W,screen_H)

fps = 25

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.zeros((W,H,3),dtype=np.uint8)

# Initialize the automaton
auto = VonNeumann((H,W),device='cuda')

updating = False
recording = False
launch_video = False

def draw_game_state(world_state,excited_state,conf_out,el_size):
    """
        Draws the game state on the screen, using the world_state array.
        For visualizing bigger pixels.

        Args:
            world_state (np.ndarray): (W,H,3) array of RGB uint8 values
            el_size (int): size of the pixels in the pygame window
        
        Returns:

    """
    W,H,_ = world_state.shape
    surf = pygame.Surface((el_size*W, el_size*H))

    for i in range(W):
        for j in range(H):
            type = world_state[i,j,0]
            if(type==1):
                surf.blit(xp, (i*el_size, j*el_size))
            elif(type==2):
                surf.blit(xm, (i*el_size, j*el_size))
            elif(type==3):
                surf.blit(yp, (i*el_size, j*el_size))
            elif(type==4):
                surf.blit(ym, (i*el_size, j*el_size))
            elif(type==5):
                surf.blit(sxp, (i*el_size, j*el_size))
            elif(type==6):
                surf.blit(sxm, (i*el_size, j*el_size))
            elif(type==7):
                surf.blit(syp, (i*el_size, j*el_size))
            elif(type==8):
                surf.blit(sym, (i*el_size, j*el_size))
            elif(type==9):
                if(conf_out[i,j,0]>0):
                    surf.blit(conf01, (i*el_size, j*el_size))
                else:
                    surf.blit(conf, (i*el_size, j*el_size))
            elif(type==10):
                surf.blit(sens0, (i*el_size, j*el_size))


            if(excited_state[i,j,0]>0):
                surf.blit(excited, (i*el_size, j*el_size))
    #PIXEL VIEWING, EFFICIENT BUT UGLY. Need el_size=1.
    # surf = pygame.surfarray.make_surface(world_state)
    # screen.blit(surf, (0,0))

    return surf

counter=0
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
        camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        auto.step()
    if(counter%100==0):
        auto.draw()
    #Retrieve the world_state from automaton
    world_state, excited_state, conf_out = auto.worldmap
    surface= draw_game_state(world_state,excited_state,conf_out,el_size)

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
    zoomed_surface = camera.apply(surface)
    screen.blit(zoomed_surface, (0, 0))
    clock.tick(fps)  # limits FPS
    curfps= clock.get_fps()
    fps_text = font.render("FPS: " + str(int(curfps)), True, (255,255,255),(0,0,0))  # Red color

    screen.blit(fps_text, (10, 10))  # Display at position (10, 10)
    # Update the screen
    pygame.display.flip()
    counter+=1



  

pygame.quit()
if(launch_video):
    video_out.release()