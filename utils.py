import os
import settings
from typing import Tuple
import numpy as np
import cv2
from imutils.perspective import four_point_transform

def save_upload_image(file_object):
    filename = file_object.filename
    
    name, extension = filename.split('.')
    
    save_filename = 'upload.' + extension
    uploaded_image_path = settings.join_path(settings.SAVE_DIR, save_filename)
    
    file_object.save(uploaded_image_path)
    
    return uploaded_image_path

def array_to_json_format(four_points):
    points_list = []
    for point in four_points:
        points_list.append({
            # Force conversion to native Python int
            'x': int(point[0]), 
            'y': int(point[1])
        })
    return points_list

class DocumentScan():
    def __init__(self):
        pass
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = 500) -> Tuple[np.ndarray, Tuple[int, int]]:
        # Get original dimensions: (height, width)
        orig_height, orig_width = image.shape[:2]
        
        # Calculate aspect ratio and new height
        aspect_ratio = orig_height / orig_width
        height = int(width * aspect_ratio)
        
        # Resize using OpenCV (interpolation=cv2.INTER_AREA is best for shrinking)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        
        return resized_image, (width, height)
    
    @staticmethod
    def apply_brigthness_constrast(input_image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
        """
        Executes a sequential photometric modulation pipeline to adjust the overall 
        luminance (brightness) and dynamic range (contrast) of an input image matrix.

        The transformations are applied as optimized linear algebraic operations 
        using the point operator formula: O(x,y) = α * I(x,y) + γ.

        Parameters:
        -----------
        input_image : np.ndarray
            The input spatial matrix (image) requiring photometric adjustment.
        brightness : int, default=0
            The scalar value dictating the magnitude of luminance translation. 
            Positive values shift the matrix towards higher intensities (lighter); 
            negative values shift it towards lower intensities (darker).
        contrast : int, default=0
            The scalar value dictating the magnitude of dynamic range scaling. 
            Positive values expand the intensity variance; negative values compress it.

        Returns:
        --------
        np.ndarray
            The photometrically modulated output image matrix.
        """
        
        # =========================================================================
        # Phase 1: Luminance (Brightness) Modulation
        # =========================================================================
        if brightness != 0:
            # Establish structural boundaries to prevent integer overflow/underflow
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
                
            # Compute the scalar multiplier (gain, α) and translation offset (bias, γ)
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            # Execute the linear transformation using optimized matrix addition
            buf = cv2.addWeighted(input_image, alpha_b, input_image, 0, gamma_b)
        else:
            # If no translation is requested, allocate a pristine copy of the input matrix
            buf = input_image.copy()

        # =========================================================================
        # Phase 2: Dynamic Range (Contrast) Scaling
        # =========================================================================
        if contrast != 0:
            # Compute the non-linear contrast correction factor.
            # This standard algorithmic formula scales the variance symmetrically 
            # around the 8-bit mid-tone intensity value (127).
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            
            # Derive the contrast gain (α) and the compensatory bias (γ)
            alpha_c = f
            gamma_c = 127 * (1 - f)

            # Execute the linear transformation to expand or compress the dynamic range
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            
        return buf
    
    def document_scanner(self, image_path):
        # =========================================================================
        # Phase 1: Image Preprocessing & Enhancement
        # =========================================================================
        # Amplify structural details to make document boundaries more prominent
        self.image = cv2.imread(image_path)
        img_resized, self.size = self.resize_image(self.image)
        filename = 'resize_image.jpg'
        RESIZE_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR, filename)
        
        cv2.imwrite(RESIZE_IMAGE_PATH, img_resized)
        
        try:
            details_img_enhance = cv2.detailEnhance(img_resized, sigma_s=20, sigma_r=0.15)    
            # Convert the enhanced BGR image to a single-channel grayscale matrix
            gray_img = cv2.cvtColor(details_img_enhance, cv2.COLOR_BGR2GRAY)   
            # Apply Gaussian smoothing to attenuate high-frequency noise and prevent false edges
            blur_img = cv2.GaussianBlur(gray_img, ksize=(5, 5), sigmaX=0)
            
            # =========================================================================
            # Phase 2: Structural Edge Detection
            # =========================================================================
            # Compute spatial gradients using the Canny algorithm to generate a binary edge map
            edges_img = cv2.Canny(blur_img, threshold1=75, threshold2=200)
            
            # =========================================================================
            # Phase 3: Morphological Transformations
            # =========================================================================
            # Define a 5x5 structural element (kernel) consisting of ones
            kernel = np.ones((5, 5), np.uint8)  
            # Dilation: Thicken the structural edges to guarantee connectivity
            dilation_img = cv2.dilate(edges_img, kernel, iterations=1) 
            # Closing: Bridge microscopic topological gaps within the continuous edge structures
            closing = cv2.morphologyEx(dilation_img, cv2.MORPH_CLOSE, kernel)
            
            # =========================================================================
            # Phase 4: Topological Contour Extraction & Heuristic Filtering
            # =========================================================================
            # Retrieve the boundary coordinates of all closed shapes within the morphological map
            contours, hire = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort the extracted contours in descending order based on their geometric area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Initialize a variable to hold the spatial coordinates of the document's corners
            four_points = None
            
            # Calculate the total pixel area of the resized image for dynamic thresholding
            total_image_area = img_resized.shape[0] * img_resized.shape[1]
            
            # Iterate through the largest contours to isolate the document quadrilateral
            for contour in contours:
                # Compute the geometric area of the current contour
                contour_area = cv2.contourArea(contour)
                
                # ---------------------------------------------------------------------
                # Heuristic 1: Area Bounding
                # Reject contours that are suspiciously large (likely the image border itself)
                # or overly small (background noise).
                # Adjust the percentages (0.70 and 0.10) based on your specific dataset.
                # ---------------------------------------------------------------------
                if contour_area > 0.70 * total_image_area or contour_area < 0.10 * total_image_area:
                    continue # Skip this contour and evaluate the next one
                
                # Compute the perimeter of the closed contour
                perimeter = cv2.arcLength(contour, True)
                
                # Execute the Douglas-Peucker algorithm to simplify the contour into a generic polygon
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Condition: If the simplified polygon contains exactly 4 vertices
                if len(approx) == 4:
                    
                    # -----------------------------------------------------------------
                    # Heuristic 2: Convexity Check
                    # A physical document must form a convex shape. If the polygon has 
                    # concave geometry ("dented" inwards), it is a background artifact.
                    # -----------------------------------------------------------------
                    if cv2.isContourConvex(approx):
                        # Flatten the structural array to a standard 2D coordinate matrix
                        four_points = np.squeeze(approx)
                        break # Optimal document boundary found; exit the loop
            
            # =========================================================================
            # Phase 5: Graphical Annotation & Rendering
            # =========================================================================
            # Safety check: Ensure the quadrilateral was successfully detected before drawing
            if four_points is not None:
                # Superimpose the identified quadrilateral boundary onto the original image (Green box)
                pass
                # cv2.drawContours(img_resized, [four_points], -1, (0, 255, 0), 3)
            
            return four_points, self.resize_image
        except:
            return None, self.size
        

    def calibrate_to_original_size(self, four_points):
        # =========================================================================
        # Phase 6: Coordinate Projection to Original Resolution
        # =========================================================================
        
        # Compute the linear scaling factor (ratio) between the original image width 
        # and the downscaled image width to determine the spatial multiplier.
        multiplier = self.image.shape[1] / self.size[0]   
        # Execute a linear transformation, applying the scalar multiplier to the 
        # coordinate matrix to project the vertices into the original resolution space.
        four_points_original_matrix = four_points * multiplier  
        # Quantize the extrapolated continuous coordinates back into discrete 
        # integer values, which is strictly required for valid matrix/pixel indexing.
        four_points_original_matrix = four_points_original_matrix.astype(int)

        # =========================================================================
        # Phase 7: Perspective Transformation (Homography)
        # =========================================================================  
        # Execute the perspective warp. 
        # This function automatically computes the destination rectangle dimensions, 
        # calculates the perspective transformation matrix, and applies the warp 
        # to project the document into a flattened, top-down orthogonal view.
        wrap_img = four_point_transform(self.image, four_points_original_matrix)
        
        # Apply brightness_constrast_image to wrap image     
        image_bright_contrast = self.apply_brigthness_constrast(wrap_img, brightness=40, contrast=60)
        
        return image_bright_contrast