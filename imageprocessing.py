import numpy as np
import cv2

class ImageProcessingLibrary:
    def __init__(self):
        pass

    def gaussian_blur(self, image, kernel_size):
        """
        Applies Gaussian blur to an image.

        Args:
            image: The input image.
            kernel_size: The size of the Gaussian kernel.

        Returns:
            The blurred image.
        """
        kernel = self.create_gaussian_kernel(kernel_size)
        blurred_image = self.convolve(image, kernel)
        return blurred_image

    def create_gaussian_kernel(self, kernel_size, sigma=1.0):
        """
        Creates a normalized Gaussian kernel.

        Args:
            kernel_size: The size of the kernel.
            sigma: The standard deviation of the Gaussian distribution.

        Returns:
            The Gaussian kernel.
        """
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size[0]), np.linspace(-1, 1, kernel_size[1]))
        distance = np.sqrt(x ** 2 + y ** 2)
        kernel = np.exp(-0.5 * (distance / sigma) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    def convolve(self, image, kernel):
        """
        Performs convolution on an image using a kernel.

        Args:
            image: The input image.
            kernel: The convolution kernel.

        Returns:
            The convolved image.
        """
        convolved_image = cv2.filter2D(image, -1, kernel)
        return convolved_image

    def canny_edge_detection(self, image, threshold1, threshold2):
        """
        Applies Canny edge detection to an image.

        Args:
            image: The input image.
            threshold1: The lower threshold value for hysteresis thresholding.
            threshold2: The upper threshold value for hysteresis thresholding.

        Returns:
            The edges detected in the image.
        """
        blurred_image = self.gaussian_blur(image, kernel_size=(5, 5))

        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
        gradient_direction = np.where(gradient_direction < 0, gradient_direction + 180, gradient_direction)

        suppressed_image = self.non_max_suppression(gradient_magnitude, gradient_direction)

        edges = self.edge_linking(suppressed_image, threshold1, threshold2)

        return edges

    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        """
        Applies non-maximum suppression to thin edges.

        Args:
            gradient_magnitude: The gradient magnitude of the image.
            gradient_direction: The gradient direction of the image.

        Returns:
            The image with non-maximum suppression applied.
        """
        suppressed_image = np.zeros_like(gradient_magnitude)
        rows, cols = gradient_magnitude.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = gradient_direction[i, j]

                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif 22.5 <= angle < 67.5:
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif 67.5 <= angle < 112.5:
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                else:
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image

    def edge_linking(self, edges, low_threshold, high_threshold):
        """
        Applies edge linking using hysteresis thresholding.

        Args:
            edges: The edge image.
            low_threshold: The lower threshold value.
            high_threshold: The upper threshold value.

        Returns:
            The edges after edge linking.
        """
        thresholded_image = np.zeros_like(edges)
        thresholded_image[(edges >= low_threshold) & (edges < high_threshold)] = 128

        strong_edges = cv2.threshold(edges, high_threshold, 255, cv2.THRESH_BINARY)[1]
        _, weak_edges = cv2.threshold(thresholded_image, 0, 255, cv2.THRESH_BINARY)
        connected_edges = cv2.Canny(weak_edges, low_threshold, high_threshold)

        edges = cv2.bitwise_or(strong_edges, connected_edges)
        return edges

    def thresholding(self, image, threshold):
        """
        Applies thresholding to an image.

        Args:
            image: The input image.
            threshold: The threshold value.

        Returns:
            The thresholded image.
        """
        binary_image = np.zeros_like(image)
        binary_image[image >= threshold] = 255
        return binary_image

    def find_contours(self, image):
        """
        Finds contours in a binary image.

        Args:
            image: The binary image.

        Returns:
            The contours found in the image.
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, image, contours):
        """
        Draws contours on an image.

        Args:
            image: The input image.
            contours: The contours to draw.

        Returns:
            The image with contours drawn.
        """
        drawn_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        return drawn_image

    def count_objects(self, image, grid_size, threshold):
        """
        Counts the number of objects in an image using grid-based analysis.

        Args:
            image: The input image.
            grid_size: The size of each grid cell.
            threshold: The threshold value for object detection.

        Returns:
            The number of objects in the image.
        """
        width, height = image.shape[:2]
        num_cells = (width // grid_size) * (height // grid_size)
        object_count = 0

        for i in range(num_cells):
            cell_image = image[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
            num_pixels = np.count_nonzero(cell_image > threshold)

            if num_pixels > 0:
                object_count += 1

        return object_count


# Usage example
image = ...  # Load or create an image
image_processor = ImageProcessingLibrary()
blurred_image = image_processor.gaussian_blur(image, kernel_size=(5, 5))
edges = image_processor.canny_edge_detection(blurred_image, threshold1=50, threshold2=150)
thresholded_image = image_processor.thresholding(image, threshold=128)
contours = image_processor.find_contours(thresholded_image)
result_image = image_processor.draw_contours(image, contours)
object_count = image_processor.count_objects(image, grid_size=10, threshold=128)

