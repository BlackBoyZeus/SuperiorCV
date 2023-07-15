class Pixel:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __repr__(self):
        return f"Pixel({self.r}, {self.g}, {self.b})"


class Layer:
    def __init__(self, image, opacity):
        self.image = image
        self.opacity = opacity

    def apply_opacity(self, pixel):
        return Pixel(
            int(pixel.r * self.opacity),
            int(pixel.g * self.opacity),
            int(pixel.b * self.opacity)
        )


class ImageEditor:
    def __init__(self, image):
        self.base_image = image
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def remove_layer(self, layer):
        self.layers.remove(layer)

    def blend_layers(self):
        height = len(self.base_image)
        width = len(self.base_image[0])
        blended_image = [[Pixel(0, 0, 0) for _ in range(width)] for _ in range(height)]

        for layer in self.layers:
            for y in range(height):
                for x in range(width):
                    base_pixel = self.base_image[y][x]
                    layer_pixel = layer.image[y][x]
                    blended_pixel = self.blend_pixels(base_pixel, layer.apply_opacity(layer_pixel))
                    blended_image[y][x] = blended_pixel

        self.base_image = blended_image

    def blend_pixels(self, pixel1, pixel2):
        return Pixel(
            min(pixel1.r + pixel2.r, 255),
            min(pixel1.g + pixel2.g, 255),
            min(pixel1.b + pixel2.b, 255)
        )

    def adjust_brightness(self, value):
        for row in self.base_image:
            for pixel in row:
                pixel.r += value
                pixel.g += value
                pixel.b += value

    def adjust_contrast(self, value):
        for row in self.base_image:
            for pixel in row:
                pixel.r = self.adjust_channel(pixel.r, value)
                pixel.g = self.adjust_channel(pixel.g, value)
                pixel.b = self.adjust_channel(pixel.b, value)

    def adjust_channel(self, channel, value):
        return min(max(int((channel - 127.5) * value + 127.5), 0), 255)

    def rotate(self, angle):
        angle %= 360
        radian = angle * 0.0174533  # Convert to radians
        height = len(self.base_image)
        width = len(self.base_image[0])
        rotated_image = [[Pixel(0, 0, 0) for _ in range(width)] for _ in range(height)]

        for y in range(height):
            for x in range(width):
                new_x = int((x - width / 2) * math.cos(radian) - (y - height / 2) * math.sin(radian) + width / 2)
                new_y = int((x - width / 2) * math.sin(radian) + (y - height / 2) * math.cos(radian) + height / 2)

                if 0 <= new_x < width and 0 <= new_y < height:
                    rotated_image[new_y][new_x] = self.base_image[y][x]

        self.base_image = rotated_image

    def crop(self, x, y, width, height):
        self.base_image = [row[x:x + width] for row in self.base_image[y:y + height]]

    def flip(self, axis):
        if axis == 0:  # Vertical flip
            self.base_image = self.base_image[::-1]
        elif axis == 1:  # Horizontal flip
            self.base_image = [row[::-1] for row in self.base_image]

    def apply_filter(self, filter_type):
        if filter_type == 'blur':
            kernel = [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
            self.base_image = self.apply_convolution(kernel)
        elif filter_type == 'sharpen':
            kernel = [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
            self.base_image = self.apply_convolution(kernel)

    def apply_convolution(self, kernel):
        height = len(self.base_image)
        width = len(self.base_image[0])
        kernel_size = len(kernel)
        padding = kernel_size // 2
        convolved_image = [[Pixel(0, 0, 0) for _ in range(width)] for _ in range(height)]

        for y in range(height):
            for x in range(width):
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        source_y = y - padding + i
                        source_x = x - padding + j

                        if 0 <= source_y < height and 0 <= source_x < width:
                            pixel = self.base_image[source_y][source_x]
                            kernel_value = kernel[i][j]
                            convolved_pixel = self.multiply_pixel(pixel, kernel_value)
                            convolved_image[y][x] = self.add_pixels(convolved_image[y][x], convolved_pixel)

        return convolved_image

    def multiply_pixel(self, pixel, value):
        return Pixel(pixel.r * value, pixel.g * value, pixel.b * value)

    def add_pixels(self, pixel1, pixel2):
        return Pixel(pixel1.r + pixel2.r, pixel1.g + pixel2.g, pixel1.b + pixel2.b)

    def apply_brush_tool(self, position, size, color):
        height = len(self.base_image)
        width = len(self.base_image[0])

        for y in range(position[1] - size, position[1] + size + 1):
            for x in range(position[0] - size, position[0] + size + 1):
                if 0 <= y < height and 0 <= x < width:
                    self.base_image[y][x] = color

    def select_region(self, start_x, start_y, end_x, end_y):
        selected_region = [row[start_x:end_x] for row in self.base_image[start_y:end_y]]
        return selected_region


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

# Rotate image
editor.rotate(30)

# Crop image
editor.crop(0, 0, 2, 2)

# Flip image
editor.flip(axis=0)

# Apply filter
editor.apply_filter('blur')

# Add layer and blend layers
layer_image = [[Pixel(100, 100, 100) for _ in range(3)] for _ in range(2)]
layer = Layer(layer_image, opacity=0.5)
editor.add_layer(layer)
editor.blend_layers()

# Apply brush tool
editor.apply_brush_tool((1, 1), size=1, color=Pixel(255, 0, 0))

# Select region
selected_region = editor.select_region(0, 0, 2, 2)
