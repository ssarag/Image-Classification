
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2


app = Flask(__name__)

@app.route('/')  # Define the root route
def index():
    return render_template('index.html')

# Load your trained CNN model
model = load_model('my_model.h5py')

# Define preprocessing function
def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to match the input size of the model
    resized_image = cv2.resize(gray_image, (150, 150))

    # Convert image to numpy array and normalize pixel values
    image_array = np.array(resized_image) / 255.0

    # Reshape the image to the required shape for CNN
    image_array = image_array.reshape(-1, 150, 150, 1)  # Adjust dimensions as needed
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    npimg = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class

    # Modify the response as per your requirement
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    result = {'predicted_class': class_names[predicted_class]}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)