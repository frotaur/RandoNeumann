from PIL import Image

def rotate_and_save_image(file_name, save_name,turns=1):
    with Image.open(file_name) as img:
        rotated = img.rotate(90*turns, expand=True)  # 90 degrees counter-clockwise
        rotated.save(save_name)

input_files = ['s','']
starter = 'xm.png'
out_fold = 'gen/'
output_files = ['xm.png','yp.png', 'xp.png', 'ym.png']

for input_file in input_files:
    input_fname = input_file+starter
    for i,output_file in enumerate(output_files):
        rotate_and_save_image(input_fname, out_fold+input_file+output_file,turns=i)