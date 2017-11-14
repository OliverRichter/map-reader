from __future__ import unicode_literals, print_function
import pyTGA


def main():
    data_bw = [
        [0, 255, 0, 0],
        [0, 0, 255, 0],
        [255, 255, 255, 0]
    ]

    data_rgb = [
        [(0, 0, 0), (255, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0)],
        [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 0)]
    ]

    data_rgba = [
        [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
        [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
        [(255, 0, 0, 150), (255, 0, 0, 150), (255, 0, 0, 150), (0, 0, 0, 0)]
    ]

    ##
    # Create from grayscale data
    image = pyTGA.Image(data=data_bw)
    # Save as TGA
    image.save("image_black_and_white")

    ##
    # Create from RGB data
    image = pyTGA.Image(data=data_rgb)
    image.save("image_rgb")

    ##
    # Create from RGBA data
    image = pyTGA.Image(data=data_rgba)
    image.save("image_rgba")

    ##
    # Save with RLE compression
    image = pyTGA.Image(data=data_rgba)
    image.save("image_rgba_compressed", compress=True)

    ##
    # Save in original format
    image = pyTGA.Image(data=data_rgba)
    image.save("image_rgba_original", original_format=True)

    ##
    # Save with 16 bit depth
    # You can start also from RGB, but you will lose data
    image = pyTGA.Image(data=data_rgb)
    image.save("test_16", force_16_bit=True)

    data_rgb_16 = [
        [(0, 0, 0), (31, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 0), (0, 0, 0), (31, 0, 0), (0, 0, 0)],
        [(31, 0, 0), (31, 0, 0), (31, 0, 0), (0, 0, 0)]
    ]

    image = pyTGA.Image(data=data_rgb_16)
    image.save("image_16_bit", force_16_bit=True)

    ##
    # Load and modify an image
    image = pyTGA.Image()
    image.load("image_black_and_white.tga").set_pixel(0, 3, 175)
    image.save("image_black_and_white_mod.tga")

    # Get some data
    print(image.get_pixel(0, 3))
    print(image.get_pixels())

if __name__ == '__main__':
    main()
