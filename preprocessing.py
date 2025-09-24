"""
OpenCV-based preprocessing for Kolam images
This module provides advanced image preprocessing to handle noise, background variation, 
and disconnections in Kolam images before feeding them to the CNN.
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Union, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KolamPreprocessor:
    """OpenCV-based preprocessor for Kolam images"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 blur_kernel_size: int = 5,
                 morph_kernel_size: int = 3,
                 threshold_method: str = 'adaptive',
                 enable_contour_detection: bool = True):
        """
        Initialize the Kolam preprocessor
        
        Args:
            target_size: Final output image size (height, width)
            blur_kernel_size: Gaussian blur kernel size (odd number)
            morph_kernel_size: Morphological operations kernel size
            threshold_method: 'adaptive', 'otsu', or 'binary'
            enable_contour_detection: Whether to detect and crop to main contour
        """
        self.target_size = target_size
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.morph_kernel_size = morph_kernel_size
        self.threshold_method = threshold_method
        self.enable_contour_detection = enable_contour_detection
        
        # Create morphological kernels
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (morph_kernel_size, morph_kernel_size))
        
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Comprehensive preprocessing pipeline for Kolam images
        
        Args:
            image_input: Path to image file, numpy array, or PIL Image
            
        Returns:
            Preprocessed image as numpy array in RGB format (H, W, 3)
        """
        try:
            # Step 1: Load image
            image = self._load_image(image_input)
            if image is None:
                raise ValueError("Failed to load image")
                
            # Step 2: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Step 3: Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
            # Step 4: Apply thresholding to binarize
            binary = self._apply_thresholding(blurred)
            
            # Step 5: Apply morphological operations to connect broken strokes
            processed = self._apply_morphological_operations(binary)
            
            # Step 6: Contour detection and cropping (optional)
            if self.enable_contour_detection:
                processed = self._crop_to_main_contour(processed)
            
            # Step 7: Resize to target size
            resized = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Step 8: Convert to 3-channel RGB format
            rgb_image = self._convert_to_rgb(resized)
            
            # Step 9: Normalize pixel values to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            # Return a black image as fallback
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
    
    def preprocess_image_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a numpy array image
        
        Args:
            image_array (np.ndarray): Input image as numpy array (RGB or BGR)
            
        Returns:
            np.ndarray: Preprocessed image as RGB array with values in [0, 1]
        """
        try:
            # Convert RGB to BGR if needed (OpenCV uses BGR)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Assume it's RGB from PIL, convert to BGR for OpenCV processing
                original = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                original = image_array.copy()
            
            logger.info(f"Processing image array with shape: {image_array.shape}")
            
            # Step 1: Convert to grayscale
            if len(original.shape) == 3:
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                gray = original.copy()
            
            # Step 2: Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
            
            # Step 3: Apply thresholding
            processed = self._apply_thresholding(blurred)
            
            # Step 4: Apply morphological operations to connect broken strokes
            processed = self._apply_morphological_operations(processed)
            
            # Step 5: Contour detection and cleanup (if enabled)
            if self.enable_contour_detection:
                processed = self._crop_to_main_contour(processed)
            
            # Step 6: Resize to target size
            resized = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Step 7: Convert to 3-channel RGB format
            rgb_image = self._convert_to_rgb(resized)
            
            # Step 8: Normalize pixel values to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in preprocessing array: {str(e)}")
            # Return a black image as fallback
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """Load image from various input types"""
        if isinstance(image_input, str):
            # Load from file path
            if not os.path.exists(image_input):
                logger.error(f"Image file not found: {image_input}")
                return None
            return cv2.imread(image_input)
            
        elif isinstance(image_input, np.ndarray):
            # Already a numpy array
            return image_input.copy()
            
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to OpenCV format
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return None
    
    def _apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding based on the selected method"""
        if self.threshold_method == 'adaptive':
            # Adaptive thresholding - good for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif self.threshold_method == 'otsu':
            # Otsu's thresholding - automatically finds optimal threshold
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Simple binary thresholding
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            
        return binary
    
    def _apply_morphological_operations(self, binary_image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to connect broken strokes"""
        # Apply morphological closing to connect nearby components
        closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Apply morphological opening to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        
        # Apply dilation to thicken lines slightly
        dilated = cv2.dilate(opened, 
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 
                           iterations=1)
        
        return dilated
    
    def _crop_to_main_contour(self, binary_image: np.ndarray) -> np.ndarray:
        """Detect main contour and crop image to focus on the kolam"""
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return binary_image
            
            # Find the largest contour (assumed to be the main kolam)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding around the bounding rectangle
            padding = 20
            h_img, w_img = binary_image.shape
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            # Crop to the padded bounding rectangle
            cropped = binary_image[y1:y2, x1:x2]
            
            # Ensure the cropped image is square by padding with black
            h_crop, w_crop = cropped.shape
            max_dim = max(h_crop, w_crop)
            
            # Create square canvas
            square_image = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Center the cropped image in the square canvas
            y_offset = (max_dim - h_crop) // 2
            x_offset = (max_dim - w_crop) // 2
            square_image[y_offset:y_offset + h_crop, x_offset:x_offset + w_crop] = cropped
            
            return square_image
            
        except Exception as e:
            logger.warning(f"Contour detection failed: {str(e)}, using original image")
            return binary_image
    
    def _convert_to_rgb(self, grayscale_image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to 3-channel RGB format"""
        if len(grayscale_image.shape) == 2:
            # Convert grayscale to RGB by stacking the same channel 3 times
            rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        else:
            # Already 3-channel, ensure it's RGB
            rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB)
            
        return rgb_image
    
    def visualize_preprocessing_steps(self, image_input: Union[str, np.ndarray, Image.Image], 
                                    save_path: Optional[str] = None) -> dict:
        """
        Visualize all preprocessing steps for debugging
        
        Args:
            image_input: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary containing images from each step
        """
        steps = {}
        
        try:
            # Original image
            original = self._load_image(image_input)
            steps['original'] = original
            
            # Grayscale
            if len(original.shape) == 3:
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                gray = original.copy()
            steps['grayscale'] = gray
            
            # Blurred
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
            steps['blurred'] = blurred
            
            # Thresholded
            binary = self._apply_thresholding(blurred)
            steps['thresholded'] = binary
            
            # Morphological operations
            morphed = self._apply_morphological_operations(binary)
            steps['morphological'] = morphed
            
            # Contour detection and cropping
            if self.enable_contour_detection:
                cropped = self._crop_to_main_contour(morphed)
                steps['cropped'] = cropped
            else:
                steps['cropped'] = morphed
            
            # Final result
            final = self.preprocess_image(image_input)
            steps['final'] = (final * 255).astype(np.uint8)
            
            # Save visualization if path provided
            if save_path:
                self._save_visualization(steps, save_path)
                
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            
        return steps
    
    def _save_visualization(self, steps: dict, save_path: str):
        """Save preprocessing visualization"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        step_names = ['original', 'grayscale', 'blurred', 'thresholded', 
                     'morphological', 'cropped', 'final']
        
        for i, step_name in enumerate(step_names):
            if i < len(axes) and step_name in steps:
                img = steps[step_name]
                if len(img.shape) == 3:
                    # Convert BGR to RGB for matplotlib
                    if step_name == 'original':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                else:
                    axes[i].imshow(img, cmap='gray')
                axes[i].set_title(step_name.title())
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(step_names), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualization saved to: {save_path}")

# Global configuration
PREPROCESS_CONFIG = {
    'PREPROCESS': True,  # Toggle preprocessing on/off
    'TARGET_SIZE': (224, 224),
    'BLUR_KERNEL_SIZE': 5,
    'MORPH_KERNEL_SIZE': 3,
    'THRESHOLD_METHOD': 'adaptive',  # 'adaptive', 'otsu', or 'binary'
    'ENABLE_CONTOUR_DETECTION': True
}

# Global preprocessor instance
_global_preprocessor = None

def get_preprocessor() -> KolamPreprocessor:
    """Get global preprocessor instance"""
    global _global_preprocessor
    if _global_preprocessor is None:
        _global_preprocessor = KolamPreprocessor(
            target_size=PREPROCESS_CONFIG['TARGET_SIZE'],
            blur_kernel_size=PREPROCESS_CONFIG['BLUR_KERNEL_SIZE'],
            morph_kernel_size=PREPROCESS_CONFIG['MORPH_KERNEL_SIZE'],
            threshold_method=PREPROCESS_CONFIG['THRESHOLD_METHOD'],
            enable_contour_detection=PREPROCESS_CONFIG['ENABLE_CONTOUR_DETECTION']
        )
    return _global_preprocessor

def preprocess_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convenience function for preprocessing images
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
        
    Returns:
        Preprocessed image as numpy array in RGB format (H, W, 3)
    """
    if not PREPROCESS_CONFIG['PREPROCESS']:
        # If preprocessing is disabled, just load and resize the image
        preprocessor = get_preprocessor()
        original = preprocessor._load_image(image_input)
        if original is None:
            return np.zeros((224, 224, 3), dtype=np.float32)
        
        # Simple resize and normalize
        if len(original.shape) == 3:
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            
        resized = cv2.resize(rgb, (224, 224))
        return resized.astype(np.float32) / 255.0
    
    preprocessor = get_preprocessor()
    return preprocessor.preprocess_image(image_input)

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("Testing Kolam Preprocessor...")
    
    # Create test image (simulate a simple kolam pattern)
    test_img = np.ones((300, 300), dtype=np.uint8) * 255  # White background
    
    # Draw some kolam-like patterns
    cv2.circle(test_img, (150, 150), 50, 0, 2)  # Central circle
    cv2.line(test_img, (50, 50), (250, 250), 0, 3)  # Diagonal line
    cv2.line(test_img, (50, 250), (250, 50), 0, 3)  # Other diagonal
    
    # Add some noise
    noise = np.random.randint(0, 50, test_img.shape, dtype=np.uint8)
    noisy_img = cv2.subtract(test_img, noise)
    
    # Test preprocessing
    preprocessed = preprocess_image(noisy_img)
    
    print(f"Original shape: {noisy_img.shape}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    print("✅ Preprocessing test completed successfully!")

if __name__ == "__main__":
    test_preprocessing()