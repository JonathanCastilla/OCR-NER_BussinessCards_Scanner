from flask import Flask, request, render_template
import numpy as np
import cv2

# =========================================================================
# Custom Module Imports
# =========================================================================
# - settings: Centralized configuration module (e.g., directory paths, environment variables).
# - utils: Utility module containing helper functions (e.g., file saving, JSON conversion).
# - NER_model_prediction: Module housing the trained spaCy Named Entity Recognition model and inference logic.
import settings
import utils
import NER_spaCy_pretrained_model as NER_model_prediction

# =========================================================================
# Application Initialization
# =========================================================================
# Instantiate the Flask WSGI application instance.
# The 'template_folder' parameter explicitly defines the directory containing HTML views.
app = Flask(__name__, template_folder='templates')

# Define a cryptographic key required for securely signing session cookies 
# (necessary for flashing messages or managing session states).
app.secret_key = 'document_scanner_app'

# Instantiate the DocumentScan class from the utility module to handle 
# computer vision operations (boundary detection, perspective transformation).
doc_scan = utils.DocumentScan()

# =========================================================================
# Route Controllers
# =========================================================================

@app.route("/", methods=['GET', 'POST'])
def scan_document():
    """
    Handles the primary entry point of the application and the initial document upload process.

    GET Request: Renders the initial document scanning interface.
    POST Request: Receives the uploaded image payload, saves it locally, executes the 
                  boundary detection algorithm (doc_scan), and returns the spatial coordinates 
                  of the document to the frontend for visualization and manual correction.
    """
    if request.method == 'POST':
        # Retrieve the binary file payload from the HTTP request.
        # 'image_name' must correspond to the HTML form's <input name="image_name">.
        file = request.files['image_name']
        
        # Execute the utility function to persist the file to the local filesystem.
        upload_image_path = utils.save_upload_image(file)
        
        # Console output for backend debugging and request tracking.
        print('==== Image saved ====')
        print(f'\n\tpath:{upload_image_path}\n')
        
        # Execute the boundary detection pipeline to isolate the document quadrilateral.
        four_points, size_img = doc_scan.document_scanner(upload_image_path)
        print(four_points)
        
        # Exception Handling: If the algorithm fails to isolate a valid 4-point contour.
        if four_points is None:
            message = 'UNABLE TO LOCATE THE COORDINATES OF DOCUMENT: points displayed are random'
            # Provide default, arbitrary coordinates to prevent frontend rendering errors.
            points = [
                {'x': 10, 'y': 10},
                {'x': 120, 'y': 10},
                {'x': 120, 'y': 120},
                {'x': 10, 'y': 120}    
            ]
            
            # Render the template, passing the failure state and default coordinates.
            return render_template('scanner.html', points=points, fileupload=True, message=message)
        
        # Standard Execution: Document successfully detected.
        else:
            # Transform the NumPy coordinate array into a JSON-compliant dictionary format.
            points = utils.array_to_json_format(four_points)
            message = 'Located the Coordinates of Document using OpenCV'
            
            # Render the template, passing the success state and calculated coordinates.
            return render_template('scanner.html', points=points, fileupload=True, message=message)
            
    # Default GET response: Serve the empty scanner view.
    return render_template('scanner.html')


@app.route("/transform", methods=['POST'])
def transform():
    """
    Executes the perspective transformation (homography) based on user-validated coordinates.

    This route expects a JSON payload containing the final (potentially user-adjusted) 
    4-point coordinates from the frontend. It applies the transformation to crop and rectify 
    the document, then saves the processed image to the disk.
    """
    try:
        # Extract the coordinate JSON payload from the asynchronous frontend request.
        points = request.json['data']
        
        # Cast the JSON array into a NumPy matrix for OpenCV compatibility.
        array = np.array(points)
        
        # Execute the perspective warp to generate an orthogonal, cropped document view.
        wrap_image = doc_scan.calibrate_to_original_size(array)
        
        # Define the output filename and absolute path for the rectified image.
        filename = "processed_image.jpg"
        processed_image_path = settings.join_path(settings.MEDIA_DIR, filename)
        
        # Persist the rectified image matrix to the local filesystem.
        cv2.imwrite(processed_image_path, wrap_image)
        
        # Return a success flag to the frontend asynchronous call.
        return 'success'
    except Exception as e:
        # Catch and handle potential errors (e.g., malformed JSON, OpenCV failures).
        print(f"Error during transformation: {e}")
        return 'failure'


@app.route("/prediction")
def prediction():
    """
    Initiates the Optical Character Recognition (OCR) and Named Entity Recognition (NER) pipeline.

    This route reads the rectified image saved by the /transform route, passes it through the 
    predictive models, and renders the extracted data entities back to the client interface.
    """
    # Construct the absolute path to the rectified image generated in the previous step.
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, "processed_image.jpg")
    
    # Load the rectified image matrix via OpenCV.
    image = cv2.imread(wrap_image_filepath)
    
    # Execute the inference pipeline to extract spatial bounding boxes and structured semantic entities.
    image_bounding_boxes, results = NER_model_prediction.get_predictions(image)
    
    # Define the output path and save the annotated image (with drawn bounding boxes) to disk.
    filename_img_bounding_boxes = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg')
    cv2.imwrite(filename_img_bounding_boxes, image_bounding_boxes)
    
    # Render the final presentation view, passing the structured extraction results to the template.
    return render_template('predictions.html', results=results)


@app.route("/about")
def about():
    """
    Renders the static informational page detailing application architecture or usage.
    """
    return render_template('about.html')


# =========================================================================
# Development Server Execution
# =========================================================================
# Conditionally initialize the Flask development server.
# Ensures the server only runs if the script is executed directly, 
# preventing accidental execution when imported as a module.
if __name__ == "__main__":
    app.run(debug=True)