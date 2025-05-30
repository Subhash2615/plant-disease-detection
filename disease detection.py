import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = tf.keras.models.load_model('/content/drive/MyDrive/finalmodel.keras', custom_objects={})
class_labels = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Tomato Mosaic Virus",
    "Target Spot",
    "Tomato Spotted Wilt Virus",
    "Tomato Yellow Leaf Curl Virus",
]

def preprocess_image(image_path, target_size=(224, 224)):

    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

image_path = '/content/drive/MyDrive/Train/bs/Bs1.jpg'

input_image = preprocess_image(image_path)

predictions = model.predict(input_image)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class index: {predicted_class}")
print(f"Class probabilities: {predictions[0]}")
print(f"Predicted class label: {class_labels[predicted_class]}")