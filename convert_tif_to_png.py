import os
from PIL import Image

def convert_tif_to_png(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            img_path = os.path.join(source_dir, filename)
            img = Image.open(img_path)
            new_filename = filename.replace('.tif', '.png')
            img.save(os.path.join(target_dir, new_filename))

if __name__ == "__main__":
    source_dir = './untested/adult'  # Change this to your source directory containing .tif files
    target_dir = 'untested_png/adult'  # Directory where .png files will be saved
    convert_tif_to_png(source_dir, target_dir)
    source_dir = './untested/child'  # Change this to your source directory containing .tif files
    target_dir = 'untested_png/child'  # Directory where .png files will be saved
    convert_tif_to_png(source_dir, target_dir)
