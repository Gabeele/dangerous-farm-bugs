import os
from flask import Flask, request, jsonify
from ml import predict_image
from werkzeug.utils import secure_filename
from io import BytesIO
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():

    print(request.files)
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        data = BytesIO()
        file.save(data)
        data.seek(0)  

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + datetime.now().strftime("%Y%m%d-%H%M%S"))
        with open(file_path, 'wb') as f:
            f.write(data.read())
        data.seek(0) 

        predicted_class_name = predict_image(data)  
        os.remove(file_path)  

        return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
