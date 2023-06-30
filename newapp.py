import torch
import os
from flask import Flask, render_template, request
from model import New_Classifier
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded!'

    file = request.files['file']
    if file.filename == '':
        return 'No file selected!'

    if file:
        # Save the uploaded file to the designated folder
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Load the model
        model = New_Classifier()
        state_dict = torch.load('checkpoint_withdropout.pth')
        model.load_state_dict(state_dict)

        # Open the uploaded file and apply transformations
        img = Image.open(file)
        img = transform(img)

        # Add a batch dimension
        #img = img.unsqueeze(0)

        # Make a forward pass through the model
        log_ps = model(img)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)

        # Get the predicted class label
        #predicted_class = top_class.item()
        #predicted_class = top_class.squeeze(0).item()
        predicted_classes = top_class.squeeze(0).tolist()

        # Get the path to the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Render the result.html template with the predicted label and image path
        return render_template('result.html', predicted_class=predicted_classes, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True)
