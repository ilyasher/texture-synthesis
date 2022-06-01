# implementation of
# https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf

import numpy as np
from PIL import Image
import time
import os
from pathlib import Path

from texturefiller import TextureFiller
from quilting      import QuiltTextureFiller

dont_blur = ['cursive.jpeg', 'text.jpeg']

wdict = {'red_abstract.jpeg': 41,
        'abstract_wall.jpeg': 67,
        'cursive.jpeg': 67,
        'aqua_wood.jpeg': 101, # bad
        'blueberries.jpeg': 101,
        'bookshelf.jpeg': 151,
        'autumn_forest.jpeg': 101,
        'stairs.jpeg': 121,
        'striped_pattern.jpeg': 101,
        'forest.jpeg': 101,
        'golden_hour_sky.jpeg': 51,
        'willow_pattern.jpeg': 101,
        'abstract_architecture.jpeg': 67,
        'grass.jpeg': 121,
        'perforated_holes.jpeg': 101,
        'dry_soil.jpeg': 67,
        'hearts.jpeg': 67,
        'almonds.jpeg': 67,
        'weathered_wood.jpeg': 41,
        'text.jpeg': 67,
        'chrisanthemum.jpeg': 101,
        'old_wood.jpeg': 121,
        'marigold_blue.jpeg': 101,
        'purple_flowers.jpeg': 67,
        'birch_trunks.jpeg': 141,
        'frozen_window.jpeg': 101,
        'flowers.jpeg': 101,
        'shells.jpeg': 101,
        'moss.jpeg': 51}

def synthesize_quilt_textures(out_h, out_w, use_small_textures=True):
    textures_dir = Path(__file__).parent / ('textures_small' if use_small_textures else 'textures')
    output_dir   = Path(__file__).parent / ('out_small' if use_small_textures else 'out')

    for filename in sorted(os.listdir(textures_dir)):

        if not (filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')):
            continue

        path = Path(__file__).parent / textures_dir / filename
        im = np.array(Image.open(path))[:, :, :3]

        if filename in wdict:
            w = wdict[filename] * np.max(im.shape[:2]) / 300
            w = int((w // 2) * 2 + 1)
        else:
            w = (np.min(im.shape[:2]) // 6) * 2 + 1

        print(f"Synthesizing {filename}, using w={w}")

        start = time.time()
        tfiller = QuiltTextureFiller(im, out_h, out_w, w=w)
        filled = tfiller.fill_texture(blur= filename not in dont_blur)
        print("Texture filling took ", time.time() - start, "seconds")

        outpath = Path(__file__).parent / output_dir / filename
        Image.fromarray(filled).save(outpath)

def fill_hole(texture_file, out_filename, w=11):
    im = np.array(Image.open(texture_file))
    im[70:130, 100:200] = 0
    valid = np.ones(shape=(im.shape[0], im.shape[1], 1))
    valid[70:130, 100:200] = 0
    Image.fromarray(im).save(Path(__file__).parent / ('hole_in_' + out_filename))
    start = time.time()
    tfiller = TextureFiller(im, valid, w=w)
    filled = tfiller.fill_texture()
    print(f"Hole filling of {texture_file} took {time.time() - start} seconds")
    Image.fromarray(filled).save(Path(__file__).parent / out_filename)

def single_pixel_texture_synthesis(texture_file, out_filename, w=11):
    im = np.array(Image.open(texture_file))[:, :, :3]

    def create_arrays_for_expanding(img, new_h, new_w):
        full = np.zeros(shape=(new_h, new_w, 3), dtype=img.dtype)
        full[:img.shape[0], :img.shape[1]] = im
        valid = np.zeros(shape=(new_h, new_w, 1), dtype=int)
        valid[:img.shape[0], :img.shape[1]] = 1
        return full, valid

    start = time.time()

    full, valid = create_arrays_for_expanding(im, int(im.shape[0]*2), int(im.shape[1]*2))
    tfiller = TextureFiller(full, valid, w=w, only_sample_from=im)
    filled = tfiller.fill_texture()

    print(f"Texture filling of {texture_file} took {time.time() - start} seconds")
    Image.fromarray(filled).save(out_filename)


if __name__ == '__main__':
    single_pixel_texture_synthesis(Path(__file__).parent / 'textures_tiny' / 'placemat.jpeg', 'placemat_bigger.jpeg', w=23)
    single_pixel_texture_synthesis(Path(__file__).parent / 'textures_tiny' / 'rocks.jpeg', 'rocks_bigger.jpeg', w=23)

    fill_hole(Path(__file__).parent / 'textures_small' / 'grass.jpeg', 'grass_filling.jpeg', w=11)
    fill_hole(Path(__file__).parent / 'textures_small' / 'bookshelf.jpeg', 'bookshelf_filling.jpeg', w=29)
    fill_hole(Path(__file__).parent / 'textures_small' / 'hearts.jpeg', 'hearts_filling.jpeg', w=29)

    synthesize_quilt_textures(600, 900, use_small_textures=True)
    synthesize_quilt_textures(1200, 1800, use_small_textures=False)

