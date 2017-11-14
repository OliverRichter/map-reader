import unittest
import os


class TestStringMethods(unittest.TestCase):

    def test_black_and_white_image(self):
        import pyTGA

        data_bw = [
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [255, 255, 255, 0]
        ]

        image = pyTGA.Image(data=data_bw)
        image.save("test_bw")

        image2 = pyTGA.Image()
        image2.load("test_bw.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_bw.tga")

    def test_RGB_image(self):
        import pyTGA

        data_rgb = [
            [(0, 0, 0), (255, 0, 0), (0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0)],
            [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgb)
        image.save("test_rgb")

        image2 = pyTGA.Image()
        image2.load("test_rgb.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_rgb.tga")

    def test_RGBA_image(self):
        import pyTGA

        data_rgba = [
            [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
            [(255, 0, 0, 150), (255, 0, 0, 150),
             (255, 0, 0, 150), (0, 0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgba)
        image.save("test_rgba")

        image2 = pyTGA.Image()
        image2.load("test_rgba.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_rgba.tga")

    def test_modify_pixel(self):
        import pyTGA

        data_rgba = [
            [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
            [(255, 0, 0, 150), (255, 0, 0, 150),
             (255, 0, 0, 150), (0, 0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgba)
        image.save("test_mod_rgba")

        image2 = pyTGA.Image(data=data_rgba)
        image2.set_pixel(0, 3, (0, 255, 0, 55))
        image2.save("test_mod_rgba_2")

        image2 = pyTGA.Image()
        image2.load("test_mod_rgba_2.tga")

        self.assertNotEqual(image.get_pixels(), image2.get_pixels())
        self.assertEqual(image.get_pixel(0, 3), (0, 0, 0, 0))
        self.assertEqual(image2.get_pixel(0, 3), (0, 255, 0, 55))

        os.remove("test_mod_rgba.tga")
        os.remove("test_mod_rgba_2.tga")

    def test_original_format(self):
        import pyTGA

        data_rgba = [
            [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
            [(255, 0, 0, 150), (255, 0, 0, 150),
             (255, 0, 0, 150), (0, 0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgba)
        image.save("test_original_rgba", original_format=True)

        image2 = pyTGA.Image()
        image2.load("test_original_rgba.tga")

        self.assertTrue(image2.is_original_format())

        os.remove("test_original_rgba.tga")

    def test_new_format(self):
        import pyTGA

        data_rgba = [
            [(0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0), (0, 0, 0, 0)],
            [(0, 0, 0, 0), (0, 0, 0, 0), (255, 0, 0, 150), (0, 0, 0, 0)],
            [(255, 0, 0, 150), (255, 0, 0, 150),
             (255, 0, 0, 150), (0, 0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgba)
        image.save("test_new_format_rgba")

        image2 = pyTGA.Image()
        image2.load("test_new_format_rgba.tga")

        self.assertFalse(image2.is_original_format())

        os.remove("test_new_format_rgba.tga")

    def test_16_bits(self):
        import pyTGA

        data_rgb = [
            [(0, 0, 0), (255, 0, 0), (0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0)],
            [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 0)]
        ]

        data_rgb_16 = [
            [(0, 0, 0), (31, 0, 0), (0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (31, 0, 0), (0, 0, 0)],
            [(31, 0, 0), (31, 0, 0), (31, 0, 0), (0, 0, 0)]
        ]

        image = pyTGA.Image(data=data_rgb)
        image.save("test_16", force_16_bit=True)

        image2 = pyTGA.Image()
        image2.load("test_16.tga")

        self.assertEqual(image2.get_pixels(), data_rgb_16)

        os.remove("test_16.tga")

    def test_data_exceptions(self):
        import pyTGA

        data_0 = [
            [0.0],
        ]

        data_1 = [
            [0],
            [0],
            [0, 1, 2]
        ]

        data_2 = [
            [()],
        ]

        data_3 = [
            [(1, 2, 3, 4, 5)],
        ]

        with self.assertRaises(pyTGA.ImageError) as img_e:
            pyTGA.Image(data=data_0)

        the_exception = img_e.exception
        self.assertEqual(the_exception.errno, -23)

        with self.assertRaises(pyTGA.ImageError) as img_e:
            pyTGA.Image(data=data_1)

        the_exception = img_e.exception
        self.assertEqual(the_exception.errno, -21)

        with self.assertRaises(pyTGA.ImageError) as img_e:
            pyTGA.Image(data=data_2)

        the_exception = img_e.exception
        self.assertEqual(the_exception.errno, -22)

        with self.assertRaises(pyTGA.ImageError) as img_e:
            pyTGA.Image(data=data_3)

        the_exception = img_e.exception
        self.assertEqual(the_exception.errno, -22)

    def test_compression_bw(self):
        import pyTGA

        data = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 255, 0, 0, 120, 200, 250],
            [0, 0, 255, 0, 0, 0, 100],
            [255, 255, 255, 0, 255, 255, 255]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_compression_bw", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_compression_bw.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_compression_bw.tga")

    def test_compression_16(self):
        import pyTGA

        data = [
            [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
             (0, 0, 0), (0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (31, 0, 0), (0, 0, 0), (0, 0, 0),
             (31, 0, 0), (31, 0, 0), (31, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (31, 0, 0), (0, 0, 0),
             (0, 0, 0), (0, 0, 0), (20, 0, 0)],
            [(31, 0, 0), (31, 0, 0), (31, 0, 0), (0, 0, 0),
             (10, 0, 0), (10, 0, 0), (10, 0, 0)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_compression_16", compress=True, force_16_bit=True)

        image2 = pyTGA.Image()
        image2.load("test_compression_16.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_compression_16.tga")

    def test_compression_RGB(self):
        import pyTGA

        data = [
            [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
             (0, 0, 0), (0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (255, 0, 0), (0, 0, 0), (0, 0, 0),
             (120, 0, 0), (200, 0, 0), (250, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0),
             (0, 0, 0), (0, 0, 0), (100, 0, 0)],
            [(255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 0, 0),
             (255, 0, 0), (255, 0, 0), (255, 0, 0)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_compression_rgb", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_compression_rgb.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_compression_rgb.tga")

    def test_compression_RGBA(self):
        import pyTGA

        data = [
            [(0, 0, 0, 255), (0, 0, 0, 255), (0, 0, 0, 255), (0, 0, 0, 255),
             (0, 0, 0, 255), (0, 0, 0, 255), (0, 0, 0, 255)],
            [(0, 0, 0, 255), (255, 0, 0, 255), (0, 0, 0, 255), (0, 0, 0, 255),
             (120, 0, 0, 255), (200, 0, 0, 255), (250, 0, 0, 255)],
            [(0, 0, 0, 255), (0, 0, 0, 255), (255, 0, 0, 255), (0, 0, 0, 255),
             (0, 0, 0, 255), (0, 0, 0, 255), (100, 0, 0, 255)],
            [(255, 0, 0, 255), (255, 0, 0, 255), (255, 0, 0, 255),
             (0, 0, 0, 255), (255, 0, 0, 255), (255, 0, 0, 255), (255, 0, 0, 255)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_compression_rgba", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_compression_rgba.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_compression_rgba.tga")

    def test_big_RLE_bw(self):
        import pyTGA

        data = [
            [0 for elm in range(500)],
            [0 for elm in range(500)],
            [0 for elm in range(500)],
            [0 for elm in range(500)],
            [0 for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_RLE_bw", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_RLE_bw.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_RLE_bw.tga")

    def test_big_pack_bw(self):
        import pyTGA

        data = [
            [elm % 255 for elm in range(500)],
            [elm % 255 for elm in reversed(range(500))],
            [elm % 255 for elm in range(500)],
            [elm % 255 for elm in reversed(range(500))],
            [elm % 255 for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_pack_bw", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_pack_bw.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_pack_bw.tga")

    def test_big_RLE_16(self):
        import pyTGA

        data = [
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_RLE_16", compress=True, force_16_bit=True)

        image2 = pyTGA.Image()
        image2.load("test_big_RLE_16.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_RLE_16.tga")

    def test_big_pack_16(self):
        import pyTGA

        data = [
            [(elm % 32, 0, 0) for elm in range(500)],
            [(elm % 32, 0, 0) for elm in reversed(range(500))],
            [(elm % 32, 0, 0) for elm in range(500)],
            [(elm % 32, 0, 0) for elm in reversed(range(500))],
            [(elm % 32, 0, 0) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_pack_16", compress=True, force_16_bit=True)

        image2 = pyTGA.Image()
        image2.load("test_big_pack_16.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_pack_16.tga")

    def test_big_RLE_rgb(self):
        import pyTGA

        data = [
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)],
            [(0, 0, 0) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_RLE_rgb", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_RLE_rgb.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_RLE_rgb.tga")

    def test_big_pack_rgb(self):
        import pyTGA

        data = [
            [(elm % 255, 0, 0) for elm in range(500)],
            [(elm % 255, 0, 0) for elm in reversed(range(500))],
            [(elm % 255, 0, 0) for elm in range(500)],
            [(elm % 255, 0, 0) for elm in reversed(range(500))],
            [(elm % 255, 0, 0) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_pack_rgb", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_pack_rgb.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_pack_rgb.tga")

    def test_big_RLE_rgba(self):
        import pyTGA

        data = [
            [(0, 0, 0, 255) for elm in range(500)],
            [(0, 0, 0, 255) for elm in range(500)],
            [(0, 0, 0, 255) for elm in range(500)],
            [(0, 0, 0, 255) for elm in range(500)],
            [(0, 0, 0, 255) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_RLE_rgba", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_RLE_rgba.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_RLE_rgba.tga")

    def test_big_pack_rgba(self):
        import pyTGA

        data = [
            [(elm % 255, 0, 0, 255) for elm in range(500)],
            [(elm % 255, 0, 0, 255) for elm in reversed(range(500))],
            [(elm % 255, 0, 0, 255) for elm in range(500)],
            [(elm % 255, 0, 0, 255) for elm in reversed(range(500))],
            [(elm % 255, 0, 0, 255) for elm in range(500)]
        ]

        image = pyTGA.Image(data=data)
        image.save("test_big_pack_rgba", compress=True)

        image2 = pyTGA.Image()
        image2.load("test_big_pack_rgba.tga")

        self.assertEqual(image.get_pixels(), image2.get_pixels())

        os.remove("test_big_pack_rgba.tga")

    def test_first_pixel_destination(self):
        import pyTGA

        data = [
            [0, 255, 0, 0],
            [0, 0, 255, 0],
            [255, 255, 255, 0]
        ]

        with self.assertRaises(pyTGA.ImageError) as img_e:
            pyTGA.Image(data=data).set_first_pixel_destination("blabla")

        the_exception = img_e.exception
        self.assertEqual(the_exception.errno, -10)

if __name__ == '__main__':
    unittest.main()
