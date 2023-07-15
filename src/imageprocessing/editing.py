import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def crop_image(self, image, x, y, width, height):
        """
        Crop the input image to the specified region of interest.

        Args:
            image: The input image as a NumPy array.
            x: The x-coordinate of the top-left corner of the region.
            y: The y-coordinate of the top-left corner of the region.
            width: The width of the region.
            height: The height of the region.

        Returns:
            The cropped image as a NumPy array.
        """
        return image[y:y+height, x:x+width]

    def resize_image(self, image, new_width, new_height):
        """
        Resize the input image to the specified dimensions.

        Args:
            image: The input image as a NumPy array.
            new_width: The desired width of the image.
            new_height: The desired height of the image.

        Returns:
            The resized image as a NumPy array.
        """
        resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                orig_i = int(i * image.shape[0] / new_height)
                orig_j = int(j * image.shape[1] / new_width)
                resized_image[i, j] = image[orig_i, orig_j]
        return resized_image

    def rotate_image(self, image, angle):
        """
        Rotate the input image by the specified angle.

        Args:
            image: The input image as a NumPy array.
            angle: The rotation angle in degrees.

        Returns:
            The rotated image as a NumPy array.
        """
        angle_rad = np.radians(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        height, width = image.shape[:2]
        new_width = int(abs(width * cos_theta) + abs(height * sin_theta))
        new_height = int(abs(width * sin_theta) + abs(height * cos_theta))
        rotated_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
        cx, cy = width // 2, height // 2  # Center coordinates
        for i in range(new_height):
            for j in range(new_width):
                x = int((j - new_width // 2) * cos_theta + (i - new_height // 2) * sin_theta) + cx
                y = int(-(j - new_width // 2) * sin_theta + (i - new_height // 2) * cos_theta) + cy
                if 0 <= x < width and 0 <= y < height:
                    rotated_image[i, j] = image[y, x]
        return rotated_image

    def flip_image(self, image, flip_code):
        """
        Flip the input image horizontally or vertically.

        Args:
            image: The input image as a NumPy array.
            flip_code: 0 for vertical flip, 1 for horizontal flip.

        Returns:
            The flipped image as a NumPy array.
        """
        if flip_code == 0:
            return image[::-1, ...]
        elif flip_code == 1:
            return image[:, ::-1, ...]
        else:
            return image

    def blend_images(self, image1, image2, alpha):
        """
        Blend two images together using alpha blending.

        Args:
            image1: The first input image as a NumPy array.
            image2: The second input image as a NumPy array.
            alpha: The blending weight, ranging from 0 to 1.

        Returns:
            The blended image as a NumPy array.
        """
        return (alpha * image1 + (1 - alpha) * image2).astype(np.uint8)

    def apply_mask(self, image, mask):
        """
        Apply a binary mask to the input image.

        Args:
            image: The input image as a NumPy array.
            mask: The binary mask image as a NumPy array.

        Returns:
            The masked image as a NumPy array.
        """
        return image * np.expand_dims(mask, axis=2)

    # Other methods (adjust_color, apply_filter, erode_image, dilate_image, etc.)
    # would require implementing custom algorithms without relying on external libraries.

# Usage example
image = ...  # Load or create an image as a NumPy array

# Initialize the ImageProcessor
processor = ImageProcessor()

# Apply pixel region manipulations
cropped_image = processor.crop_image(image, x=100, y=100, width=300, height=300)
resized_image = processor.resize_image(image, new_width=500, new_height=500)
rotated_image = processor.rotate_image(image, angle=45)
flipped_image = processor.flip_image(image, flip_code=0)
image2 = ...  # Load or create a second image as a NumPy array
blended_image = processor.blend_images(image, image2, alpha=0.5)
mask = ...  # Create a binary mask as a NumPy array
masked_image = processor.apply_mask(image, mask)
