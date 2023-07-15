class Pixel:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __repr__(self):
        return f"Pixel({self.r}, {self.g}, {self.b})"


class ImageEditor:
    def __init__(self, image):
        self.image = image

    def adjust_brightness(self, value):
        for row in self.image:
            for pixel in row:
                pixel.r += value
                pixel.g += value
                pixel.b += value

    def adjust_contrast(self, value):
        for row in self.image:
            for pixel in row:
                pixel.r = int((pixel.r - 127.5) * value + 127.5)
                pixel.g = int((pixel.g - 127.5) * value + 127.5)
                pixel.b = int((pixel.b - 127.5) * value + 127.5)

    def adjust_saturation(self, value):
        for row in self.image:
            for pixel in row:
                hsv = self.rgb_to_hsv(pixel)
                hsv.s *= value
                rgb = self.hsv_to_rgb(hsv)
                pixel.r = rgb.r
                pixel.g = rgb.g
                pixel.b = rgb.b

    def rgb_to_hsv(self, pixel):
        r, g, b = pixel.r / 255, pixel.g / 255, pixel.b / 255
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        if delta == 0:
            h = 0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)
        else:
            h = 60 * (((r - g) / delta) + 4)

        if cmax == 0:
            s = 0
        else:
            s = delta / cmax

        v = cmax

        return Pixel(int(h), int(s * 100), int(v * 100))

    def hsv_to_rgb(self, pixel):
        h, s, v = pixel.r, pixel.g / 100, pixel.b / 100

        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return Pixel(int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# Usage example
pixels = [
    [Pixel(255, 0, 0), Pixel(0, 255, 0), Pixel(0, 0, 255)],
    [Pixel(255, 255, 255), Pixel(128, 128, 128), Pixel(0, 0, 0)],
]
editor = ImageEditor(pixels)

# Adjust brightness
editor.adjust_brightness(50)

# Adjust contrast
editor.adjust_contrast(1.5)

# Adjust saturation
editor.adjust_saturation(1.2)
