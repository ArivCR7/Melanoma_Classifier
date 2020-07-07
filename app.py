from flask import Flask
from flask import request, render_template
import os
import predict_melanoma
import torch

app = Flask(__name__)
UPLOAD_FOLDER = "/home/ariv/pets/melanoma/static/"

@app.route('/', methods=['POST','GET'])
def upload_predict():
    if request.method=='POST':
        image_file = request.files['file']
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            prediction = predict_melanoma.predict(image_path, model)
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=0)

if __name__ == '__main__':
    model = predict_melanoma.SEResnext50_32x4d(pretrained=None, wp=torch.tensor(0))
    print(f"Loading from model: melanoma_model.bin")
    model.load_state_dict(torch.load("model/melanoma_model.bin", map_location='cpu'))
    model.eval()

    app.run(port=5000, debug=True)