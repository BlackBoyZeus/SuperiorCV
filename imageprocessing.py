import numpy as np

class ImageProcessingLibrary:
    def __init__(self):
        pass

    def gaussian_blur(self, image, kernel_size):
        kernel = self.create_gaussian_kernel(kernel_size)
        blurred_image = self.convolve(image, kernel)
        return blurred_image

    def create_gaussian_kernel(self, kernel_size, sigma=1.0):
        kx = np.linspace(-(kernel_size[0] // 2), kernel_size[0] // 2, kernel_size[0])
        ky = np.linspace(-(kernel_size[1] // 2), kernel_size[1] // 2, kernel_size[1])
        xx, yy = np.meshgrid(kx, ky)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        return kernel

    def convolve(self, image, kernel):
        height, width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape[:2]
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        convolved_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                patch = padded_image[i:i+kernel_height, j:j+kernel_width]
                result = np.sum(patch * kernel)
                convolved_image[i, j] = result
        return convolved_image

    def canny_edge_detection(self, image, threshold1, threshold2):
        # Implement Canny edge detection algorithm
        # Step 1: Apply Gaussian blur to reduce noise
        blurred_image = self.gaussian_blur(image, kernel_size=(3, 3))
        
        # Step 2: Compute gradients using Sobel operator
        gradient_x = self.convolve(blurred_image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        gradient_y = self.convolve(blurred_image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        
        # Step 3: Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
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

                # Compare gradient magnitude with neighboring pixels along the direction of the gradient
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

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image

    def edge_linking(self, edges, low_threshold, high_threshold):
        # Implement edge linking algorithm (Hysteresis thresholding)
        rows, cols = edges.shape
        strong_edges = np.zeros_like(edges)
        weak_edges = np.zeros_like(edges)

        strong_edges[edges >= high_threshold] = 255
        weak_edges[(edges >= low_threshold) & (edges < high_threshold)] = 255

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if weak_edges[i, j] == 255:
                    if np.max(strong_edges[i-1:i+2, j-1:j+2]) == 255:
                        strong_edges[i, j] = 255

        return strong_edges

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
blurred_image = image_processor.gaussian_blur(image, kernel_size=(3, 3))
gradient_magnitude, gradient_direction = image_processor.calculate_gradients(blurred_image)
suppressed_image = image_processor.non_max_suppression(gradient_magnitude, gradient_direction)
edges = image_processor.edge_linking(suppressed_image, low_threshold=50, high_threshold=150)
thresholded_image = image_processor.thresholding(image, threshold=128)
contours = image_processor.find_contours(thresholded_image)
result_image = image_processor.draw_contours(image, contours)
processed_image = image_processor.new_method(image)
