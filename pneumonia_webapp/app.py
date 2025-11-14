from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Configure upload settings - use absolute path
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static folder configuration
app.static_folder = 'static'

# For a different model name:
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models','pneumonia_model_final.h5')


def load_model():
    try:
        model_path = app.config['MODEL_PATH']
        logging.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at {model_path}"
            logging.error(error_msg)
            return None, error_msg
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Log model information
        input_shape = model.input_shape
        logging.info(f"Model loaded successfully. Input shape: {input_shape}")
        
        # Verify input shape
        if input_shape[1:] != (224, 224, 3):
            error_msg = f"Unexpected model input shape: {input_shape}. Expected (None, 224, 224, 3)"
            logging.error(error_msg)
            return None, error_msg
        
        return model, None
            
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logging.error(error_msg)
        return None, error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error in load_model: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

# Initialize model
model, model_error = load_model()
if model_error:
    logging.error(f"Initial model loading failed: {model_error}")
else:
    logging.info("Model initialized successfully")

# Add route to reload model
@app.route('/reload_model')
def reload_model():
    global model, model_error
    model, model_error = load_model()
    return jsonify({
        "status": "success" if model is not None else "error",
        "message": "Model reloaded successfully" if model is not None else model_error
    })

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to match the model's expected input size (224x224)
        img = cv2.resize(img, (224, 224))
        logging.info("Resized image to 224x224")
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logging.info(f"Preprocessed image shape: {img.shape}")
        return img, None
    except Exception as e:
        logging.error(f"Error in preprocess_image: {str(e)}")
        return None, str(e)

def is_xray_image(image_path):
    """
    Strict validation function to determine if an image is a chest X-ray
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Unable to read image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Initialize validation metrics
        validation_passed = 0
        validation_required = 6  # Increased from 5 to 6 out of 7 tests
        failure_reasons = []
        
        # TEST 1: Basic image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        logging.info(f"X-ray validation - brightness: {mean_val:.2f}, contrast: {std_val:.2f}")
        
        # Stricter brightness range (X-rays typically have moderate brightness)
        if 60 < mean_val < 170:  # Narrowed from 50-180 to 60-170
            validation_passed += 1
        else:
            failure_reasons.append(f"Brightness {mean_val:.2f} outside X-ray range (60-170)")
        
        # Higher contrast requirement
        if std_val > 50:  # Increased from 45 to 50
            validation_passed += 1
        else:
            failure_reasons.append(f"Contrast {std_val:.2f} too low for X-ray (min 50)")
        
        # TEST 2: Edge detection for bone structures
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        edge_mean = np.mean(sobel_mag)
        edge_std = np.std(sobel_mag)
        logging.info(f"X-ray validation - edge mean: {edge_mean:.2f}, edge std: {edge_std:.2f}")
        
        if edge_mean > 25:  # Increased from 20 to 25
            validation_passed += 1
        else:
            failure_reasons.append(f"Edge definition {edge_mean:.2f} too low for X-ray (min 25)")
        
        if edge_std > 30:  # Increased from 25 to 30
            validation_passed += 1
        else:
            failure_reasons.append(f"Edge variation {edge_std:.2f} too low for X-ray (min 30)")
        
        # TEST 3: Color check - X-rays should have minimal color variation
        color_std = np.std(img, axis=(0, 1))
        color_variation = np.max(color_std) - np.min(color_std)
        logging.info(f"X-ray validation - color variation: {color_variation:.2f}")
        
        if color_variation < 15:  # More strict - reduced from 20 to 15
            validation_passed += 1
        else:
            failure_reasons.append(f"Color variation {color_variation:.2f} too high for X-ray (max 15)")
        
        # TEST 4: Histogram distribution check
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        logging.info(f"X-ray validation - histogram std: {hist_std:.2f}")
        
        if hist_std > 700:  # Increased from 600 to 700
            validation_passed += 1
        else:
            failure_reasons.append(f"Histogram variation {hist_std:.2f} too low for X-ray (min 700)")
        
        # TEST 5: Symmetry check (chest X-rays have some bilateral symmetry)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        if left_half.shape != right_half_flipped.shape:
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32)))
        symmetry_ratio = symmetry_diff / mean_val
        logging.info(f"X-ray validation - symmetry ratio: {symmetry_ratio:.4f}")
        
        if symmetry_ratio < 0.25:  # More strict - reduced from 0.3 to 0.25
            validation_passed += 1
        else:
            failure_reasons.append(f"Symmetry ratio {symmetry_ratio:.4f} too high for chest X-ray (max 0.25)")
        
        # Final decision - more strict validation
        logging.info(f"X-ray validation - passed {validation_passed}/7 tests (requires {validation_required})")
        
        # Return results
        if validation_passed >= validation_required:
            logging.info("Image validated as X-ray")
            return True, None
        else:
            failure_msg = "Not a chest X-ray: " + "; ".join(failure_reasons)
            logging.warning(failure_msg)
            return False, failure_msg
            
    except Exception as e:
        error_msg = f"Error in X-ray validation: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

def get_prediction(img):
    try:
        # Input validation
        if model is None:
            raise ValueError("Model is not loaded")
        
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Invalid input type. Expected numpy array, got {type(img)}")
            
        logging.info(f"Input image shape: {img.shape}, dtype: {img.dtype}")
        logging.info(f"Input value range: min={np.min(img):.3f}, max={np.max(img):.3f}")
        
        # Make prediction with error catching
        try:
            logging.info("Starting prediction...")
            pred = model.predict(img, batch_size=1)
            logging.info(f"Raw prediction output: shape={pred.shape}, values={pred}")
            
            if pred.size == 0:
                raise ValueError("Model returned empty prediction")
                
            pred_value = float(pred[0][0])
            
        except Exception as pred_error:
            logging.error(f"Prediction failed: {str(pred_error)}")
            raise ValueError(f"Model prediction failed: {str(pred_error)}")
        
        # Validate prediction value
        if not (0 <= pred_value <= 1):
            logging.warning(f"Unexpected prediction value: {pred_value}, clipping to [0,1]")
            pred_value = np.clip(pred_value, 0, 1)
        
        # FIXED: Better confidence calculation
        if pred_value > 0.5:
            # Pneumonia prediction
            diagnosis = "PNEUMONIA"
            confidence = pred_value * 100  # Use raw prediction as confidence
        else:
            # Normal prediction  
            diagnosis = "NORMAL"
            confidence = (1 - pred_value) * 100  # Use inverse for normal
        
        result = {
            "diagnosis": diagnosis,
            "confidence": min(float(confidence), 100.0),
            "raw_score": float(pred_value)
        }
        
        logging.info(f"Prediction successful: {result}")
        return result, None
        
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}\nFull error: {repr(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Log request information
        logging.info("Received prediction request")
        logging.info(f"Request files: {request.files.keys() if request.files else 'No files'}")
        logging.info(f"Request content type: {request.content_type}")
        
        # Check model status
        if model is None:
            logging.error("Prediction attempted with unloaded model")
            try:
                return render_template('error.html', error="Model not loaded. Please try again later."), 500
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Error: Model not loaded. Please try again later.", 500

        # Validate file upload
        if 'file' not in request.files:
            logging.warning("No file part in request")
            try:
                return render_template('error.html', error="No file uploaded"), 400
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Error: No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("Empty filename submitted")
            try:
                return render_template('error.html', error="No file selected"), 400
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Error: No file selected", 400

        if not allowed_file(file.filename):
            logging.warning(f"Invalid file type: {file.filename}")
            try:
                return render_template('error.html', error="Please upload a JPG or PNG image"), 400
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Error: Please upload a JPG or PNG image", 400

        # Save the uploaded file with error handling
        try:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"

            # Use the proper UPLOAD_FOLDER path instead of hardcoded 'static'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            file.save(file_path)
            logging.info(f"File saved successfully: {file_path}")
            logging.info(f"Saved file size: {os.path.getsize(file_path)} bytes")
            
            # Verify file was saved and is readable
            if not os.path.exists(file_path):
                raise IOError("File was not saved successfully")
                
            file_size = os.path.getsize(file_path)
            logging.info(f"Saved file size: {file_size} bytes")
            
            if file_size == 0:
                raise IOError("Saved file is empty")
                
        except Exception as save_error:
            error_msg = f"Error saving file: {str(save_error)}"
            logging.error(error_msg, exc_info=True)
            try:
                return render_template('error.html', error="Could not save uploaded file"), 500
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Error: Could not save uploaded file", 500

        # Validate if the image is an X-ray
        is_xray, xray_error = is_xray_image(file_path)
        if not is_xray:
            logging.warning(f"Non-X-ray image detected: {xray_error}")
            try:
                return render_template('error.html', error="Invalid image. Please upload a chest X-ray image."), 400
            except Exception as template_error:
                logging.error(f"Error rendering error template: {template_error}")
                return "Invalid image. Please upload a chest X-ray image.", 400

        # Process the image with detailed error handling
        try:
            img, error = preprocess_image(file_path)
            if error:
                error_msg = f"Could not process image: {error}"
                logging.error(f"Image preprocessing failed: {error}")
                try:
                    return render_template('error.html', error=error_msg), 500
                except Exception as template_error:
                    logging.error(f"Error rendering error template: {template_error}")
                    return f"Error: {error_msg}", 500
                
            if img is None:
                raise ValueError("Preprocessing returned None without an error message")
                
            logging.info(f"Image preprocessed successfully: shape={img.shape}, dtype={img.dtype}")
                
        except Exception as prep_error:
            logging.error(f"Unexpected error in image preprocessing: {str(prep_error)}", exc_info=True)
            return render_template('error.html', error=f"Image processing error: {str(prep_error)}"), 500

        # Get prediction with detailed error handling
        try:
            result, error = get_prediction(img)
            if error:
                logging.error(f"Prediction failed: {error}")
                return render_template('error.html', error=f"Could not make prediction: {error}"), 500
                
            if result is None:
                raise ValueError("Prediction returned None without an error message")
                
            logging.info("Prediction completed successfully")
                
        except Exception as pred_error:
            logging.error(f"Unexpected error in prediction: {str(pred_error)}", exc_info=True)
            return render_template('error.html', error=f"Prediction error: {str(pred_error)}"), 500

        # Get patient info
        name = request.form.get('patientName', 'Not provided')
        age = request.form.get('patientAge', 'Not provided')
        gender = request.form.get('patientGender', 'Not provided')

        # Add current datetime for the template
        current_time = datetime.now()
        
        # Return results with all required variables
        try:
            # Create web-accessible path for the image
            relative_image_path = f"uploads/{filename}"
            
            return render_template('result.html',
                                prediction=result['diagnosis'],
                                confidence=result['confidence'],
                                image_path=relative_image_path,
                                name=name,
                                age=age,
                                gender=gender,
                                now=current_time)  # Pass the datetime object
        except Exception as template_error:
            error_msg = f"Error rendering result template: {str(template_error)}"
            logging.error(error_msg, exc_info=True)
            try:
                return render_template('error.html', 
                                    error="Error displaying results. Please try again."), 500
            except Exception as error_template_error:
                logging.error(f"Error rendering error template: {error_template_error}")
                return "Error displaying results. Please try again.", 500

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logging.error(error_msg, exc_info=True)
        try:
            return render_template('error.html', 
                                error="An error occurred processing your request. Please try again."), 500
        except Exception as template_error:
            logging.error(f"Error rendering error template: {template_error}")
            return "An error occurred processing your request. Please try again.", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        return "File not found", 404

@app.route('/info')
def server_info():
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        hostname = "Unknown"
        local_ip = "Unable to determine"
    
    return jsonify({
        "server_status": "running",
        "model_status": "loaded" if model is not None else "not loaded",
        "hostname": hostname,
        "local_ip": local_ip,
        "endpoints": {
            "home": "/",
            "predict": "/predict (POST)",
            "health": "/health",
            "info": "/info",
            "reload_model": "/reload_model"
        },
        "versions": {
            "tensorflow": tf.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat()
    })
if __name__ == "__main__":
    # Set up logging (file only, no console spam)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log')
        ],
        force=True
    )
    
    # Simple startup message
    print("🏥 Pneumonia Detection Server")
    print("=" * 40)
    
    # Check model
    model_path = app.config['MODEL_PATH']
    if not os.path.exists(model_path):
        print("❌ Model not found")
        logging.error(f"Model file not found at {model_path}")
    else:
        print("✅ Model loaded")
        logging.info(f"Model file found at {model_path}")
    
    # Create required directories
    models_dir = os.path.dirname(model_path)
    os.makedirs(models_dir, exist_ok=True)
    
    # Server configuration
    host = '0.0.0.0'
    port = 5000
    
    # Get local IP (simplified)
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "localhost"
    
    # Minimal server info
    print(f"🌐 Server: http://localhost:{port}")
    if local_ip != "localhost":
        print(f"🌐 Network: http://{local_ip}:{port}")
    print("=" * 40)
    print("Starting server... Press Ctrl+C to stop")
    
    # Log to file only
    logging.info("Server starting")
    logging.info(f"Host: {host}, Port: {port}")
    logging.info(f"Local IP: {local_ip}")
    
    # Start server
    try:
        app.run(debug=True, host=host, port=port, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        logging.info("Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Server error: {e}")


