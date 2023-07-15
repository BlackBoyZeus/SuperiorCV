import numpy as np
import cv2

class ImageProcessingLibrary:
    def __init__(self):
        pass

    def gaussian_blur(self, image, kernel_size):
        kernel = self.create_gaussian_kernel(kernel_size)
        blurred_image = self.convolve(image, kernel)
        return blurred_image

    def create_gaussian_kernel(self, kernel_size, sigma=1.0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size[0]), np.linspace(-1, 1, kernel_size[1]))
        distance = np.sqrt(x**2 + y**2)
        kernel = np.exp(-0.5 * (distance / sigma)**2)
        kernel /= np.sum(kernel)
        return kernel

    def convolve(self, image, kernel):
        convolved_image = cv2.filter2D(image, -1, kernel)
        return convolved_image

    def canny_edge_detection(self, image, threshold1, threshold2):
        # Step 1: Apply Gaussian blur to reduce noise
        blurred_image = self.gaussian_blur(image, kernel_size=(5, 5))

        # Step 2: Compute gradients using Sobel operator
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Step 3: Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
        gradient_direction = np.where(gradient_direction < 0, gradient_direction + 180, gradient_direction)

        # Step 4: Apply non-maximum suppression to thin edges
        suppressed_image = self.non_max_suppression(gradient_magnitude, gradient_direction)

        # Step 5: Apply double thresholding and edge linking
        edges = self.edge_linking(suppressed_image, threshold1, threshold2)

        return edges

    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        # Implement non-maximum suppression algorithm
        suppressed_image = np.zeros_like(gradient_magnitude)
        rows, cols = gradient_magnitude.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = gradient_direction[i, j]

                # Determine the two neighboring pixels to compare the gradient magnitude with
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                elif 22.5 <= angle < 67.5:
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                elif 67.5 <= angle < 112.5:
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                else:
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                # Suppress the pixel if it is not the maximum among the neighboring pixels along the gradient direction
                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image

    def edge_linking(self, edges, low_threshold, high_threshold):
        # Implement edge linking algorithm (Hysteresis thresholding)
        thresholded_image = np.zeros_like(edges)
        thresholded_image[(edges >= low_threshold) & (edges < high_threshold)] = 128

        strong_edges = cv2.threshold(edges, high_threshold, 255, cv2.THRESH_BINARY)[1]
        _, weak_edges = cv2.threshold(thresholded_image, 0, 255, cv2.THRESH_BINARY)
        connected_edges = cv2.Canny(weak_edges, low_threshold, high_threshold)

        edges = cv2.bitwise_or(strong_edges, connected_edges)
        return edges

    def thresholding(self, image, threshold):
        # Implement thresholding algorithm
        binary_image = np.zeros_like(image)
        binary_image[image >= threshold] = 255
        return binary_image

    def find_contours(self, image):
        # Implement contour detection algorithm
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, image, contours):
        # Implement contour drawing algorithm
        drawn_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        return drawn_image

    def new_method(self, image):
        # Implement your own innovative image processing algorithm
        # ...

        return image

# Usage example
image = ...  # Load or create an image
image_processor = ImageProcessingLibrary()
blurred_image = image_processor.gaussian_blur(image, kernel_size=(5, 5))
edges = image_processor.canny_edge_detection(blurred_image, threshold1=50, threshold2=150)
thresholded_image = image_processor.thresholding(image, threshold=128)
contours = image_processor.find_contours(thresholded_image)
result_image = image_processor.draw_contours(image, contours)
processed_image = image_processor.new_method(image)
