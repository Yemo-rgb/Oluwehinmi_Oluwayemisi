import os
import sqlite3
import numpy as np
from flask import Flask, request, render_template, g
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# --- Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
DATABASE = 'database.db'

# --- Load the Trained Model ---
try:
    model = load_model('face_emotionModel.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the emotions
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Database Functions ---

def get_db():
    """Get a database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize the database and create the table if it doesn't exist."""
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                image_data BLOB NOT NULL,
                detected_emotion TEXT NOT NULL
            )
        ''')
        db.commit()
        print("Database table 'users' initialized.")

# --- Helper Function ---

def preprocess_image(image_path):
    """Preprocess the image for the model."""
    # Load the image in grayscale (color_mode='grayscale')
    img = load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    
    # Convert image to array
    img_array = img_to_array(img)
    
    # Normalize the image (same as in training)
    img_array = img_array / 255.0
    
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # 1. Get form data
            name = request.form['name']
            email = request.form['email']
            
            if 'image' not in request.files or not request.files['image'].filename:
                return "No image file selected.", 400
            
            image_file = request.files['image']
        
        except KeyError as e:
            print(f"Form data missing: {e}")
            return f"Form error: missing {e}. Please go back and try again.", 400
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An unexpected error occurred.", 500

        # --- Handle Image ---
        
        # 1. Save the file temporarily
        filename = secure_filename(image_file.filename)
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            image_file.save(temp_image_path)

            # 2. Preprocess the image *from the saved path*
            processed_img = preprocess_image(temp_image_path)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error processing the uploaded image. Is it a valid image file?", 400
        
        # 3. Read the file's data for the database
        with open(temp_image_path, 'rb') as f:
            image_data_blob = f.read()
            
        # 4. Clean up the temporary file
        os.remove(temp_image_path)
        
        # 5. Make a prediction
        if model is None:
            return "Error: Model is not loaded.", 500
            
        prediction = model.predict(processed_img)
        
        # 6. Get the result
        emotion_index = np.argmax(prediction)
        detected_emotion = EMOTIONS[emotion_index]

        # 7. Save to database
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, image_data, detected_emotion) VALUES (?, ?, ?, ?)",
                (name, email, image_data_blob, detected_emotion)
            )
            db.commit()
        except Exception as e:
            print(f"Database error: {e}")
            return "Error saving to database.", 500

        # 8. Create the user-friendly response
        response_message = f"You are {detected_emotion.lower()}."
        
        if detected_emotion == 'Sad':
            response_message = "You are frowning. Why are you sad?"
        elif detected_emotion == 'Angry':
            response_message = "You look angry. What's wrong?"
        elif detected_emotion == 'Happy':
            response_message = "You look happy! That's great to see."
        elif detected_emotion == 'Surprise':
            response_message = "You look surprised!"
        
        # Return the response
        return f"<h1>Thank you, {name}!</h1><p>{response_message}</p><a href='/'>Go Back</a>"

    # If it's a GET request, just show the form
    return render_template('index.html')

# --- Run the App ---
if __name__ == '__main__':
    # Create the 'static/uploads' directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    # Initialize the database
    init_db()
    
    # Run the app
    app.run(debug=True, port=8080)