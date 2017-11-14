# pyTGA
A pure Python module to manage **TGA** *images*. This module is compatible with **Python 2** and **Python 3** and we refer on the *New TGA Format*.

The library supports these kind of formats (compressed with *RLE* or uncompressed):

* Grayscale - 8 bit depth
* RGB - 16 bit depth
* RGB - 24 bit depth
* RGBA - 32 bit depth

As you can see in the example you can use the python basic types for data.

## Install

Simply type:
```bash
python setup.py install
```

## Test

```bash
cd test
python test_module.py
```

## Contributing

Contributions are welcome, so please feel free to fix bugs, improve things, provide documentation. For anything submit a personal message or fork the project to make a pull request and so on... thanks!

## Example

```python
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
```
