from __future__ import unicode_literals, print_function
from sys import version_info
from copy import deepcopy
from struct import pack, unpack

__all__ = ["Image", "ImageError", "VERSION"]


VERSION = "1.0.2"


def dec_byte(data, size=1, littleEndian=True):
    """Decode some data from bytes.

    Args:
        data (bytes): data to decode
        size (int): number of bites of the data
        littleEndian (bool): little endian or big endian

    Returns:
        int: the decoded data
    """
    order = '<' if littleEndian else '>'
    format_ = (None, 'B', 'H', None, 'I')[size]

    return unpack(order + format_, data)[0]


def multiple_dec_byte(stream, num, size=1, littleEndian=True):
    """Decode multiple data of the same size from a file.

    Args:
        stream (file): an IO file stream
        num (int): number of same data to decode
        size (int): size in bytes of the data
        littleEndian (bool): little endian or big endian

    Returns:
        list[num]: all the data decoded
    """
    return [dec_byte(stream.read(size), size, littleEndian) for number in range(num)]


def gen_byte(data, size=1, littleEndian=True):
    """Generate bytes from data.

    Args:
        data (int): data to encode
        size (int): size in bytes of the data
        littleEndian (bool): little endian or big endian

    Returns:
        bytes[size]: conversion of the data in bytes
    """
    order = '<' if littleEndian else '>'
    format_ = (None, 'B', 'H', None, 'I')[size]

    return pack(order + format_, data)


def gen_pixel_rgba(c_r, c_g, c_b, alpha=None):
    """Generate an RGBA pixel:

    Args:
        c_r (int): red color value (0-255)
        c_g (int): green color value (0-255)
        c_b (int): blue color value (0-255)
        alpha (int): transparency of the color (0-255, default: None)

    Returns:
        bytes[3-4]: the conversion of the color in bytes

        If alpha is None the result color will be an RGB.
    """
    if alpha is not None:
        return pack('<I', alpha << 24 | c_r << 16 | c_g << 8 | c_b)
    else:
        return pack('<BBB', c_b, c_g, c_r)


def gen_pixel_rgb_16(c_r, c_g, c_b):
    """Generate an RGB pixel of 16 bit:

    Args:
        c_r (int): red color value (0-31)
        c_g (int): green color value (0-31)
        c_b (int): blue color value (0-31)

    Returns:
        bytes[2]: the conversion of the color in 2 bytes (16 bit)

        If the ranges of the color are greater then 31 the information will be
        cutted.
    """
    mask = 0b11111
    tmp = 0b0
    tmp |= (c_r & mask) << 11
    tmp |= (c_g & mask) << 6
    tmp |= (c_b & mask) << 1
    ##
    # alpha is not useful but
    # is inserted like the standard whant
    # - 1 : is visible
    # - 0 : is not visible
    #
    tmp |= 0b1

    return pack('<H', tmp)


def get_rgb_from_16(data):
    """Construct an RGB color from 16 bit of data.

    Args:
        second_byte (bytes): the first bytes read
        first_byte (bytes): the second bytes read

    Returns:
        tuple(int, int, int): the RGB color
    """
    # Args are inverted because is little endian
    c_r = (data & 0b1111100000000000) >> 11
    c_g = (data & 0b0000011111000000) >> 6
    c_b = (data & 0b111110) >> 1

    return (c_r, c_g, c_b)


class TGAHeader(object):

    """Header object for TGA files."""

    def __init__(self):
        """Initialize all fields.

        Here we have some details for each field:

        ## Field(1)
        # ID LENGTH (1 byte):
        #   Number of bites of field 6, max 255.
        #   Is 0 if no image id is present.
        #
        ## Field(2)
        # COLOR MAP TYPE (1 byte):
        #   - 0 : no color map included with the image
        #   - 1 : color map included with the image
        #
        ## Field(3)
        # IMAGE TYPE (1 byte):
        #   - 0  : no data included
        #   - 1  : uncompressed color map image
        #   - 2  : uncompressed true color image
        #   - 3  : uncompressed black and white image
        #   - 9  : run-length encoded color map image
        #   - 10 : run-length encoded true color image
        #   - 11 : run-length encoded black and white image
        #
        ## Field(4)
        # COLOR MAP SPECIFICATION (5 bytes):
        #   - first_entry_index (2 bytes) : index of first color map entry
        #   - color_map_length  (2 bytes)
        #   - color_map_entry_size (1 byte)
        #
        ##  Field(5)
        # IMAGE SPECIFICATION (10 bytes):
        #   - x_origin  (2 bytes)
        #   - y_origin  (2 bytes)
        #   - image_width   (2 bytes)
        #   - image_height  (2 bytes)
        #   - pixel_depht   (1 byte):
        #       - 8 bit  : grayscale
        #       - 16 bit : RGB (5-5-5-1) bit per color
        #                  Last one is alpha (visible or not)
        #       - 24 bit : RGB (8-8-8) bit per color
        #       - 32 bit : RGBA (8-8-8-8) bit per color
        #   - image_descriptor (1 byte):
        #       - bit 3-0 : number of attribute bit per pixel
        #       - bit 5-4 : order in which pixel data is transferred
        #                   from the file to the screen
        #  +-----------------------------------+-------------+-------------+
        #  | Screen destination of first pixel | Image bit 5 | Image bit 4 |
        #  +-----------------------------------+-------------+-------------+
        #  | bottom left                       |           0 |           0 |
        #  | bottom right                      |           0 |           1 |
        #  | top left                          |           1 |           0 |
        #  | top right                         |           1 |           1 |
        #  +-----------------------------------+-------------+-------------+
        #       - bit 7-6 : must be zero to insure future compatibility
        #
        """
        # Field(1)
        self.id_length = 0
        # Field(2)
        self.color_map_type = 0
        # Field(3)
        self.image_type = 0
        # Field(4)
        self.first_entry_index = 0
        self.color_map_length = 0
        self.color_map_entry_size = 0
        # Field(5)
        self.x_origin = 0
        self.y_origin = 0
        self.image_width = 0
        self.image_height = 0
        self.pixel_depht = 0
        self.image_descriptor = 0

    def to_bytes(self):
        """Convert the object to bytes.

        Returns:
            bytes: the conversion in bytes"""
        tmp = bytearray()

        tmp += gen_byte(self.id_length)
        tmp += gen_byte(self.color_map_type)
        tmp += gen_byte(self.image_type)
        tmp += gen_byte(self.first_entry_index, 2)
        tmp += gen_byte(self.color_map_length, 2)
        tmp += gen_byte(self.color_map_entry_size)
        tmp += gen_byte(self.x_origin, 2)
        tmp += gen_byte(self.y_origin, 2)
        tmp += gen_byte(self.image_width, 2)
        tmp += gen_byte(self.image_height, 2)
        tmp += gen_byte(self.pixel_depht)
        tmp += gen_byte(self.image_descriptor)

        return tmp


class TGAFooter(object):

    """Footer object for TGA files."""

    def __init__(self):
        """Initialize all fields."""
        self.extension_area_offset = 0  # 4 bytes
        self.developer_directory_offset = 0  # 4 bytes
        self.__signature = bytes(
            bytearray("TRUEVISION-XFILE".encode('ascii')))  # 16 bytes
        self.__dot = bytes(bytearray('.'.encode('ascii')))  # 1 byte
        self.__end = bytes(bytearray([0]))  # 1 byte

    def to_bytes(self):
        """Convert the object to bytes.

        Returns:
            bytes: the conversion in bytes
        """
        tmp = bytearray()

        tmp += gen_byte(self.extension_area_offset, 4)
        tmp += gen_byte(self.developer_directory_offset, 4)
        tmp += self.__signature
        tmp += self.__dot
        tmp += self.__end

        return tmp


class ImageError(Exception):

    """Error of the Image class."""

    def __init__(self, msg, errname):
        super(ImageError, self).__init__(msg)
        error_map = {
            'pixel_dest_position': -10,
            'bad_row_length': -21,
            'bad_pixel_length': -22,
            'bad_pixel_value': -23,
            'non_supported_type': -31,
        }
        self.errno = error_map.get(errname, None)


class Image(object):

    """Main object to manage TGA images."""

    def __init__(self, data=None):
        """Initialize the image.

        Args:
            data (list of list): data is an array of array that contains pixels.
                For more details on data go to 'check' function.

        Returns:
            Image

        """
        self._pixels = None

        if data is not None:
            self.check(data)
            self._pixels = deepcopy(data)

        # Screen destination of first pixel
        self.__bottom_left = 0b0
        self.__bottom_right = 0b1 << 4
        self.__top_left = 0b1 << 5
        self.__top_right = 0b1 << 4 | 0b1 << 5

        # Default values
        self._first_pixel = self.__top_left
        self._header = TGAHeader()
        self._footer = TGAFooter()
        self.__new_TGA_format = True

    @staticmethod
    def check(data):
        """Control if data are a valid list of pixels.

        Pixels can be of 3 types:
            * black and white -> int
            * RGB -> (int, int, int)
            * RGBA -> (int, int, int, int)

        Args:
            data (list of list): data is an array of array that contains pixels.

        Example:
            data = [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]
            ]

            data = [
                [(42, 0, 0), (0, 0, 0), (160, 0, 0)],
                [(0, 0, 0), (0, 10, 0), (0, 0, 0)],
                [(0, 0, 255), (0, 0, 0), (0, 0, 0)],
            ]

        Returns:
            None

        Raises:
            ImageError
        """
        tmp_len = len(data[0])
        row_num = 0
        for row in data:
            row_num += 1
            if len(row) != tmp_len:
                raise ImageError(
                    "row number {0} has different length from first row".format(
                        row_num),
                    'bad_row_length'
                )
            for pixel in row:
                if type(pixel) == tuple:
                    if len(pixel) < 3 or len(pixel) > 4:
                        raise ImageError(
                            "'{0}' is not a valid pixel tuple".format(pixel),
                            'bad_pixel_length'
                        )
                elif type(pixel) != int:
                    raise ImageError(
                        "'{0}' is not a valid pixel value".format(pixel),
                        'bad_pixel_value'
                    )

    def set_first_pixel_destination(self, dest):
        """Change the destination of first pixel.

        Possible values:
        * 'bl' (string): bottom left
        * 'br' (string): bottom right
        * 'tl' (string): top left
        * 'tr' (string): top right

        Args:
            dest (string): destination of the first pixel

        Returns:
            Image

        Raises:
            ImageError
        """
        if dest.lower() == 'bl':
            self._first_pixel = self.__bottom_left
            return self
        elif dest.lower() == 'br':
            self._first_pixel = self.__bottom_right
            return self
        elif dest.lower() == 'tl':
            self._first_pixel = self.__top_left
            return self
        elif dest.lower() == 'tr':
            self._first_pixel = self.__top_right
            return self
        else:
            raise ImageError(
                "'{0}' is not a valid pixel destination".format(dest),
                'pixel_dest_position'
            )

    def is_original_format(self):
        """Control if the image is old style.

        Returns:
            bool: if the image is in new TGA format
        """
        return not self.__new_TGA_format

    def set_pixel(self, row, col, value):
        """Change a pixel.

        Args:
            row (int): number of the row (starts from 0)
            col (int): number of the column (starts from 0)
            value (int-tuple): the pixel value. See 'check' function for more
                detail son pixels.

        Returns:
            Image
        """
        self._pixels[row][col] = value
        return self

    def get_pixel(self, row, col):
        """Retreive a pixel.

        Args:
            row (int): number of the row (starts from 0)
            col (int): number of the column (starts from 0)

        Returns:
            int-tuple: the pixel selected. See 'check' function for more
                detail son pixels.
        """
        return self._pixels[row][col]

    def get_pixels(self):
        """Extract data.

        Returns:
            list: all the pixels of the image. See 'check' function for more
                detail son pixels.
        """
        return self._pixels

    def load(self, file_name):
        """Open a TGA image.

        Args:
            file_name (string): the name of the TGA image

        Returns:
            Image

        Raises:
            ImageError
        """
        with open(file_name, "rb") as image_file:
            # Check footer
            image_file.seek(-26, 2)
            self._footer.extension_area_offset = dec_byte(
                image_file.read(4), 4)
            self._footer.developer_directory_offset = dec_byte(
                image_file.read(4), 4)
            signature = image_file.read(16)
            dot = image_file.read(1)
            zero = dec_byte(image_file.read(1))

            if signature == "TRUEVISION-XFILE".encode('ascii') and\
                    dot == ".".encode('ascii') and zero == 0:
                self.__new_TGA_format = True
            else:
                self.__new_TGA_format = False

            # Read Header
            image_file.seek(0)
            # ID LENGTH
            self._header.id_length = dec_byte(image_file.read(1))
            # COLOR MAP TYPE
            self._header.color_map_type = dec_byte(image_file.read(1))
            # IMAGE TYPE
            self._header.image_type = dec_byte(image_file.read(1))
            # COLOR MAP SPECIFICATION
            self._header.first_entry_index = dec_byte(image_file.read(2), 2)
            self._header.color_map_length = dec_byte(image_file.read(2), 2)
            self._header.color_map_entry_size = dec_byte(image_file.read(1))
            # IMAGE SPECIFICATION
            self._header.x_origin = dec_byte(image_file.read(2), 2)
            self._header.y_origin = dec_byte(image_file.read(2), 2)
            self._header.image_width = dec_byte(image_file.read(2), 2)
            self._header.image_height = dec_byte(image_file.read(2), 2)
            self._header.pixel_depht = dec_byte(image_file.read(1))
            self._header.image_descriptor = dec_byte(image_file.read(1))

            self._pixels = []
            if self._header.image_type == 2 or\
                    self._header.image_type == 3:
                for row in range(self._header.image_height):
                    self._pixels.append([])
                    for col in range(self._header.image_width):
                        if self._header.image_type == 3:
                            self._pixels[row].append(
                                dec_byte(image_file.read(1)))
                        elif self._header.image_type == 2:
                            if self._header.pixel_depht == 16:
                                self._pixels[row].append(
                                    get_rgb_from_16(dec_byte(image_file.read(2), 2)))
                            elif self._header.pixel_depht == 24:
                                c_b, c_g, c_r = multiple_dec_byte(
                                    image_file, 3)
                                self._pixels[row].append((c_r, c_g, c_b))
                            elif self._header.pixel_depht == 32:
                                c_b, c_g, c_r, alpha = multiple_dec_byte(
                                    image_file, 4)
                                self._pixels[row].append(
                                    (c_r, c_g, c_b, alpha))
                        else:
                            raise ImageError(
                                "type num '{0}'' is not supported".format(
                                    self._header.image_type),
                                'non_supported_type'
                            )
            ##
            # Decode
            #
            elif self._header.image_type == 10 or\
                    self._header.image_type == 11:
                self._pixels.append([])
                tot_pixels = self._header.image_height * self._header.image_width
                pixel_count = 0
                while pixel_count != tot_pixels:
                    if len(self._pixels[-1]) == self._header.image_width:
                        self._pixels.append([])
                    repetition_count = dec_byte(image_file.read(1))
                    RLE = (repetition_count & 0b10000000) >> 7 == 1
                    count = (repetition_count & 0b01111111) + 1
                    pixel_count += count
                    if RLE:
                        pixel = None
                        if self._header.image_type == 11:
                            pixel = dec_byte(image_file.read(1))
                        elif self._header.image_type == 10:
                            if self._header.pixel_depht == 16:
                                pixel = get_rgb_from_16(
                                    dec_byte(image_file.read(2), 2))
                            elif self._header.pixel_depht == 24:
                                c_b, c_g, c_r = multiple_dec_byte(
                                    image_file, 3)
                                pixel = (c_r, c_g, c_b)
                            elif self._header.pixel_depht == 32:
                                c_b, c_g, c_r, alpha = multiple_dec_byte(
                                    image_file, 4)
                                pixel = (c_r, c_g, c_b, alpha)
                        else:
                            raise ImageError(
                                "type num '{0}'' is not supported".format(
                                    self._header.image_type),
                                'non_supported_type'
                            )
                        for num in range(count):
                            self._pixels[-1].append(pixel)
                    else:
                        for num in range(count):
                            if self._header.image_type == 11:
                                self._pixels[-1].append(
                                    dec_byte(image_file.read(1)))
                            elif self._header.image_type == 10:
                                if self._header.pixel_depht == 16:
                                    self._pixels[-1].append(
                                        get_rgb_from_16(dec_byte(image_file.read(2), 2)))
                                elif self._header.pixel_depht == 24:
                                    c_b, c_g, c_r = multiple_dec_byte(
                                        image_file, 3, 1)
                                    self._pixels[-1].append((c_r, c_g, c_b))
                                elif self._header.pixel_depht == 32:
                                    c_b, c_g, c_r, alpha = multiple_dec_byte(
                                        image_file, 4, 1)
                                    self._pixels[-1].append(
                                        (c_r, c_g, c_b, alpha))
                            else:
                                raise ImageError(
                                    "type num '{0}'' is not supported".format(
                                        self._header.image_type),
                                    'non_supported_type'
                                )
        return self

    def save(self, file_name, original_format=False, force_16_bit=False,
             compress=False):
        """Save the image as a TGA file.

        Args:
            file_name (string): the name with which you want to save
            original_format (bool): save or not in olt TGA format (< 2.0)
            force_16_bit (bool): save the image with 16 bit depth
            compress (bool): compress the image with RLE or not

        Returns:
            Image

        """
        # ID LENGTH
        self._header.id_length = 0
        # COLOR MAP TYPE
        self._header.color_map_type = 0
        # COLOR MAP SPECIFICATION
        self._header.first_entry_index = 0
        self._header.color_map_length = 0
        self._header.color_map_entry_size = 0
        # IMAGE SPECIFICATION
        self._header.x_origin = 0
        self._header.y_origin = 0
        self._header.image_width = len(self._pixels[0])
        self._header.image_height = len(self._pixels)
        self._header.image_descriptor = 0b0 | self._first_pixel

        ##
        # IMAGE TYPE
        # IMAGE SPECIFICATION (pixel_depht)
        tmp_pixel = self._pixels[0][0]
        if type(tmp_pixel) == int:
            self._header.image_type = 3
            self._header.pixel_depht = 8
        elif type(tmp_pixel) == tuple:
            self._header.image_type = 2
            if len(tmp_pixel) == 3:
                if not force_16_bit:
                    self._header.pixel_depht = 24
                else:
                    self._header.pixel_depht = 16
            elif len(tmp_pixel) == 4:
                self._header.pixel_depht = 32

        if compress:
            if self._header.image_type == 3:
                self._header.image_type = 11
            elif self._header.image_type == 2:
                self._header.image_type = 10

        with open("{0:s}.tga".format(file_name), "wb") as image_file:
            image_file.write(self._header.to_bytes())

            if not compress:
                for row in self._pixels:
                    for pixel in row:
                        if self._header.image_type == 3:
                            image_file.write(gen_byte(pixel))
                        elif self._header.image_type == 2:
                            if self._header.pixel_depht == 16:
                                image_file.write(gen_pixel_rgb_16(*pixel))
                            elif self._header.pixel_depht == 24:
                                image_file.write(gen_pixel_rgba(*pixel))
                            elif self._header.pixel_depht == 32:
                                image_file.write(gen_pixel_rgba(*pixel))
            else:
                for row in self._pixels:
                    for repetition_count, pixel_value in self._encode(row):
                        image_file.write(gen_byte(repetition_count))
                        if repetition_count > 127:
                            if self._header.image_type == 11:
                                image_file.write(gen_byte(pixel_value))
                            elif self._header.image_type == 10:
                                if self._header.pixel_depht == 16:
                                    image_file.write(
                                        gen_pixel_rgb_16(*pixel_value))
                                elif self._header.pixel_depht == 24:
                                    image_file.write(
                                        gen_pixel_rgba(*pixel_value))
                                elif self._header.pixel_depht == 32:
                                    image_file.write(
                                        gen_pixel_rgba(*pixel_value))
                        else:
                            for pixel in pixel_value:
                                if self._header.image_type == 11:
                                    image_file.write(gen_byte(pixel))
                                elif self._header.image_type == 10:
                                    if self._header.pixel_depht == 16:
                                        image_file.write(
                                            gen_pixel_rgb_16(*pixel))
                                    elif self._header.pixel_depht == 24:
                                        image_file.write(
                                            gen_pixel_rgba(*pixel))
                                    elif self._header.pixel_depht == 32:
                                        image_file.write(
                                            gen_pixel_rgba(*pixel))

            if self.__new_TGA_format and not original_format:
                image_file.write(self._footer.to_bytes())

        return self

    @staticmethod
    def _encode(row):
        """Econde a row of pixels.

        This function is a generator used during the compression phase. More
        information on packets generated are after returns section.

        Args:
            row (list): a list of pixels. See 'check' function for more details

        Returns:
            tuple: repetition_count and pixel_value of the current compression
                in the given row

        ##
        # Run-length encoded (RLE) images comprise two types of data
        # elements:Run-length Packets and Raw Packets.
        #
        # The first field (1 byte) of each packet is called the
        # Repetition Count field. The second field is called the
        # Pixel Value field. For Run-length Packets, the Pixel Value
        # field contains a single pixel value. For Raw
        # Packets, the field is a variable number of pixel values.
        #
        # The highest order bit of the Repetition Count indicates
        # whether the packet is a Raw Packet or a Run-length
        # Packet. If bit 7 of the Repetition Count is set to 1, then
        # the packet is a Run-length Packet. If bit 7 is set to
        # zero, then the packet is a Raw Packet.
        #
        # The lower 7 bits of the Repetition Count specify how many
        # pixel values are represented by the packet. In
        # the case of a Run-length packet, this count indicates how
        # many successive pixels have the pixel value
        # specified by the Pixel Value field. For Raw Packets, the
        # Repetition Count specifies how many pixel values
        # are actually contained in the next field. This 7 bit value
        # is actually encoded as 1 less than the number of
        # pixels in the packet (a value of 0 implies 1 pixel while a
        # value of 0x7F implies 128 pixels).
        #
        # Run-length Packets should never encode pixels from more than
        # one scan line. Even if the end of one scan
        # line and the beginning of the next contain pixels of the same
        # value, the two should be encoded as separate
        # packets. In other words, Run-length Packets should not wrap
        # from one line to another. This scheme allows
        # software to create and use a scan line table for rapid, random
        # access of individual lines. Scan line tables are
        # discussed in further detail in the Extension Area section of
        # this document.
        #
        #
        # Pixel format data example:
        #
        # +=======================================+
        # | Uncompressed pixel run                |
        # +=========+=========+=========+=========+
        # | Pixel 0 | Pixel 1 | Pixel 2 | Pixel 3 |
        # +---------+---------+---------+---------+
        # | 144     | 144     | 144     | 144     |
        # +---------+---------+---------+---------+
        #
        # +==========================================+
        # | Run-length Packet                        |
        # +============================+=============+
        # | Repetition Count           | Pixel Value |
        # +----------------------------+-------------+
        # | 1 bit |       7 bit        |             |
        # +----------------------------|     144     |
        # |   1   |  3 (num pixel - 1) |             |
        # +----------------------------+-------------+
        #
        # +====================================================================================+
        # | Raw Packet                                                                         |
        # +============================+=============+=============+=============+=============+
        # | Repetition Count           | Pixel Value | Pixel Value | Pixel Value | Pixel Value |
        # +----------------------------+-------------+-------------+-------------+-------------+
        # | 1 bit |       7 bit        |             |             |             |             |
        # +----------------------------|     144     |     144     |     144     |     144     |
        # |   0   |  3 (num pixel - 1) |             |             |             |             |
        # +----------------------------+-------------+-------------+-------------+-------------+
        #
        """
        repetition_count = None
        pixel_value = None
        ##
        # States:
        # - 0: init
        # - 1: run-length packet
        # - 2: raw packet
        #
        state = 0
        index = 0

        while index != len(row):
            if state == 0:
                repetition_count = 0
                if index == len(row) - 1:
                    pixel_value = [row[index]]
                    yield (repetition_count, pixel_value)
                elif row[index] == row[index + 1]:
                    repetition_count |= 0b10000000
                    pixel_value = row[index]
                    state = 1
                else:
                    pixel_value = [row[index]]
                    state = 2
                index += 1
            elif state == 1 and row[index] == pixel_value:
                if repetition_count & 0b1111111 == 127:
                    yield (repetition_count, pixel_value)
                    repetition_count = 0b10000000
                else:
                    repetition_count += 1
                index += 1
            elif state == 2 and row[index] != pixel_value:
                if repetition_count & 0b1111111 == 127:
                    yield (repetition_count, pixel_value)
                    repetition_count = 0
                    pixel_value = [row[index]]
                else:
                    repetition_count += 1
                    pixel_value.append(row[index])
                index += 1
            else:
                yield (repetition_count, pixel_value)
                state = 0

        if state != 0:
            yield (repetition_count, pixel_value)
