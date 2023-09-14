# Von Neumann cellular automaton

This is a simple pygame+pytorch program implementing the Von Neumann cellular automaton. Using GPU acceleration, it can simulate worlds upwards of 10 Million cells, although the visualization for such huge words is not yet possible.


## How to use :

First run `pip install -r requirements.txt` to install the required libraries.

On top of main.py, choose the size of the world by setting the value to W and H. Choose also "el_size", the size of the representation of each cell (9 is a good number for this). This will generate a world with W*H cells, each cell represented by a 9\*9 image, resulting in a (9*H,9*W) window. Make sur this fits on your screen resolution.

You can now launch the program, it will start with a random configuration. You can interact with the world in the following way :

- 'spacebar' : start/stop time evolution
- 'o' : reset to random initial state
- 'e' : inject excitations to half the cells
- 'r' : toggle recording. When exiting the program, a video will be saved in `Videos/`, stitching together all recorded moments. A small red dot appear in the bottom right of the screen to indicate recording in action.
- Using the scrolling-wheel will zoom in and out. While zoomed in, you can click and drag to move around the space. Zoom and movements are for now not saved on the recording, but that might change in the future.