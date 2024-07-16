import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('cifar10_model.h5')

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Threshold for confidence
confidence_threshold = 0.5

def load_image(filename):
    img = Image.open(filename)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def classify_image():
    global img_label, confidence_label, logits_label
    filename = filedialog.askopenfilename()
    if filename:
        img_array = load_image(filename)
        prediction = model.predict(img_array)
        logits = prediction[0]
        
        probabilities = softmax(logits)
        
        print("Logits:", logits)
        
        for class_name, logit in zip(class_names, logits):
            print(f"{class_name}: {logit:.2f}")
        
        class_idx = np.argmax(probabilities)
        class_name = class_names[class_idx]
        confidence = probabilities[class_idx]
        
        img = Image.open(filename)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        
        img_label.config(image=img)
        img_label.image = img
        
        if confidence > confidence_threshold:
            result_label.config(text=f"Đây là: {class_name} (Confidence: {confidence:.2f})")
        else:
            result_label.config(text="Không có lớp này trong bộ dữ liệu")
        
      
        confidence_text = "\n".join([f"{class_names[i]}: {prob:.2f}" for i, prob in enumerate(probabilities)])
        confidence_label.config(text=confidence_text)

root = tk.Tk()
root.title("Phân loại ảnh CIFAR-10")

frame = tk.Frame(root)
frame.pack(pady=20)

btn_select = tk.Button(frame, text="Chọn ảnh", command=classify_image)
btn_select.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(frame, text="Chưa có kết quả", font=('Arial', 14))
result_label.pack(side=tk.LEFT, padx=10)

img_label = tk.Label(root)
img_label.pack(pady=20)

confidence_label = tk.Label(root, text="", font=('Arial', 12))
confidence_label.pack(pady=10)

root.mainloop()
