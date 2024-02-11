import pygame
from Camera import Camera
from Automaton import *
import cv2, torch, numpy as np, os
from torchenhanced.util import showTens
# TODO : Migrate all this into a big class. That way the helper methods are easier to write.


# Initialize the pygame screen 
pygame.init()
el_size = 9
W,H = 64,64

font = pygame.font.SysFont(None, 25) 
graph_folder = 'new_graph/'

# Load the images for the automaton
raw_images = {}
raw_images['xm'] = pygame.image.load(graph_folder+'xm.png')
raw_images['xp'] = pygame.image.load(graph_folder+'xp.png')
raw_images['ym'] = pygame.image.load(graph_folder+'ym.png')
raw_images['yp'] = pygame.image.load(graph_folder+'yp.png')

raw_images['sxp'] = pygame.image.load(graph_folder+'sxp.png')
raw_images['sxm'] = pygame.image.load(graph_folder+'sxm.png')
raw_images['syp'] = pygame.image.load(graph_folder+'syp.png')
raw_images['sym'] = pygame.image.load(graph_folder+'sym.png')

raw_images['conf'] = pygame.image.load(graph_folder+'conf0.png')
raw_images['conf01'] = pygame.image.load(graph_folder+'conf01.png')
raw_images['sens0'] = pygame.image.load(graph_folder+'sens0.png')
raw_images['excited'] = pygame.image.load(graph_folder+'excited.png')

images=[0]
for key,value in raw_images.items():
    exec(f'{key} = pygame.transform.scale(raw_images["{key}"], (el_size, el_size))')
# xm = pygame.transform.scale(pygame.image.load(graph_folder+'xm.png'), (el_size, el_size))
# xp = pygame.transform.scale(pygame.image.load(graph_folder+'xp.png'), (el_size, el_size))
# ym = pygame.transform.scale(pygame.image.load(graph_folder+'ym.png'), (el_size, el_size))
# yp = pygame.transform.scale(pygame.image.load(graph_folder+'yp.png'), (el_size, el_size))

# sxm = pygame.transform.scale(pygame.image.load(graph_folder+'sxm.png'), (el_size, el_size))
# sxp = pygame.transform.scale(pygame.image.load(graph_folder+'sxp.png'), (el_size, el_size))
# sym = pygame.transform.scale(pygame.image.load(graph_folder+'sym.png'), (el_size, el_size))
# syp = pygame.transform.scale(pygame.image.load(graph_folder+'syp.png'), (el_size, el_size))

# conf = pygame.transform.scale(pygame.image.load(graph_folder+'conf0.png'), (el_size, el_size))
# sens0 = pygame.transform.scale(pygame.image.load(graph_folder+'sens0.png'), (el_size, el_size))
# excited = pygame.transform.scale(pygame.image.load(graph_folder+'excited.png'), (el_size, el_size))
# conf01 = pygame.transform.scale(pygame.image.load(graph_folder+'conf01.png'), (el_size, el_size))

screen_W, screen_H = W*el_size, H*el_size

screen = pygame.display.set_mode((screen_W,screen_H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(screen_W,screen_H)


fps = 60

#Initialize the world_state array, of size (W,H,3) of RGB values at each position.
world_state = np.zeros((W,H,3),dtype=np.uint8)

device='cpu'
# Initialize the automaton
auto = BoolVonNeumann((H,W),device=device)
# auto.set_state(torch.ones((1,H,W)),excitations=torch.ones((1,H,W)).to(torch.bool))
state_opti = torch.load(os.path.join('states','best_state.pt'),map_location=device)
excitations = torch.load(os.path.join('states','initial_excitation.pt'),map_location=device)

auto.set_state(state_opti[None],excitations=excitations)
# auto.set_state(torch.zeros_like(state_opti[None]),excitations=excitations)


# uncomment for svaed state:
# state = torch.load('save_state.pt',map_location=device)
# excitations = torch.load('save_excitation.pt',map_location=device)
# auto.set_state(state,excitations=excitations)

#Uncomment for replicator
# state = torch.zeros_like(auto.births)
# state[:,2:70,5:210] = torch.load(os.path.join('states','repli.pt'),map_location=device)[None,:,:]
# state[:,70:75,5+34] =4
# excitations = torch.zeros_like(auto.excitations)
# excitations[:,2:70,5:210]=torch.load(os.path.join('states','repli_exci.pt'),map_location=device)[None,:,:]
# auto.set_state(state,excitations=excitations)

updating = False
recording = False
launch_video = False

blit_dict = {1:xp,2:xm,3:yp,4:ym,5:sxp,6:sxm,7:syp,8:sym,9:conf,10:sens0}
blit_keys = {1:'xp',2:'xm',3:'yp',4:'ym',5:'sxp',6:'sxm',7:'syp',8:'sym',9:'conf',10:'sens0'}
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
            status = world_state[i,j,0]
            if(status!=9 and status!=0):
                surf.blit(blit_dict[status], (i*el_size, j*el_size))
            elif(status==9) :
                if(conf_out[i,j,0]>0):
                    surf.blit(conf01, (i*el_size, j*el_size))
                else:
                    surf.blit(conf, (i*el_size, j*el_size))


            if(excited_state[i,j,0]>0):
                surf.blit(excited, (i*el_size, j*el_size))
    #PIXEL VIEWING, EFFICIENT BUT UGLY. Need el_size=1.
    # surf = pygame.surfarray.make_surface(world_state)
    # screen.blit(surf, (0,0))

    return surf

counter=0
erasing = False
state = 0
while running:
    for event in pygame.event.get():
        # Event loop. Here we deal with all the interactivity
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if(event.key == pygame.K_ESCAPE):
                running = False
            if(event.key == pygame.K_SPACE):
                updating=not updating
            if(event.key == pygame.K_r):
                recording= not recording
            if(event.key ==pygame.K_o):
                auto.reset_state()
            if(event.key == pygame.K_e):
                auto.inj_excitations()
            if(event.key == pygame.K_k):
                auto.is_killed = torch.ones_like(auto.is_killed)
                auto.kill_dead()
            if(event.key == pygame.K_RIGHT):
                auto.step()
            if(event.key == pygame.K_s):
                statio = auto.get_state()
                torch.save(statio,'save_state.pt')
                torch.save(auto.excitations,'save_excitation.pt')
            if(event.key == pygame.K_BACKSPACE):
                erasing=not erasing
            if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                state = int(event.unicode)

        elif event.type == pygame.MOUSEBUTTONDOWN:
        # Check if the left mouse button was clicked
            x, y = camera.convert_mouse_pos(event.pos)
            x = int(x//el_size)
            y = int(y//el_size)
            if event.button == 3:
                if(not erasing):
                    auto.excitations[:,y,x] = ~auto.excitations[:,y,x]
                else:
                    auto.is_killed[:,max(y-5,0):min(y+5,H),max(x-5,0):min(x+5,W)] = 1
                    auto.kill_dead()
            if event.button == 1:
                status = auto.get_state()
                status[:,y,x] = state
                auto.set_state(status,excitations=auto.excitations)
                # auto.births[:,y,x] = state
                # auto.make_births()

        # Handle the event loop for the camera (disabled for now)
        camera.handle_event(event)
    
    if(updating):
        # Step the automaton if we are updating
        auto.step()

    auto.draw()
    #Retrieve the world_state from automaton
    world_state, excited_state, conf_out = auto.worldmap
    surface= draw_game_state(world_state,excited_state,conf_out,el_size)
    if(state!=0):
        xmove,ymove=camera.convert_mouse_pos(pygame.mouse.get_pos())
        surface.blit(raw_images[blit_keys[state]], (0, 0))
        surface.blit(blit_dict[state], (xmove//el_size*el_size, ymove//el_size*el_size))
    #For recording
    if(recording):
        if(not launch_video):
            launch_video = True
            os.makedirs('Videos',exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            numvids = len(os.listdir('Videos/'))
            vid_loc = f'Videos/Neu_{numvids}.mp4'
            video_out = cv2.VideoWriter(vid_loc, fourcc, 60.0, (screen_W, screen_H))

        # Convert Pygame surface to a string buffer
        frame_str = pygame.image.tostring(surface, "RGB")

        # Convert this string buffer to a numpy array
        frame_np = np.frombuffer(frame_str, dtype=np.uint8).reshape((screen_H,screen_W,3))


        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)


        video_out.write(frame_bgr)
        pygame.draw.circle(surface, (255,0,0), (screen_W-10,screen_H-10),2)
    
    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window (disabled for now)
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0, 0))

    clock.tick(fps)  # limits FPS
    curfps= clock.get_fps()
    fps_text = font.render("FPS: " + str(int(curfps)), True, (255,255,255),(0,0,0))  # Red color

    # screen.blit(fps_text, (10, 10))  # Display at position (10, 10)
    # Update the screen
    pygame.display.flip()
    counter+=1


if(launch_video):
    print('release kraken')
    video_out.release()

pygame.quit()
