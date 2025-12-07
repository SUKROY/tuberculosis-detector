from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tb_cnn_lightweight.tflite")
interpreter.allocate_tensors()

# Get input-output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))   # <-- DISINI
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return "No file uploaded", 400

    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    img_array = preprocess_image(upload_path)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    result = "TUBERCULOSIS" if prediction > 0.5 else "NORMAL"

    return render_template(
        "result.html",
        result=result,
        image_path="/" + upload_path
    )


if __name__ == "__main__":
    from flask import Flask
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

