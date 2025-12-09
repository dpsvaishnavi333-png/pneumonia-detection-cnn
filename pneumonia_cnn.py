# ✅ Pneumonia Detection (Final & Balanced Version)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Define dataset paths
train_dir = r"C:\Users\sinch\Downloads\chest_xray\train"
val_dir = r"C:\Users\sinch\Downloads\chest_xray\val"
test_dir = r"C:\Users\sinch\Downloads\chest_xray\test"

print("✅ Dataset paths set correctly!")

# Step 2: Data augmentation & normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print("✅ Images loaded successfully!")

# Step 3: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Step 4: Train the model (no class weights needed)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    verbose=1
)

# Step 5: Save trained model
model.save(r"C:\Users\sinch\Desktop\pneumonia_model.keras")
print("✅ Model saved successfully!")

# Step 6: Plot accuracy & loss graphs
plt.figure(figsize=(10, 4))

# Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# 🌟 Advanced Pneumonia Detection Result Dashboard 🌟
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ Load the trained model
model = load_model(r"C:\Users\sinch\Desktop\pneumonia_model.keras")
print("✅ Model loaded successfully!")

# ✅ Choose image path
img_path = r"C:\Users\sinch\Downloads\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"

# ✅ Preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ✅ Predict
raw_pred = model.predict(img_array)
prediction = raw_pred[0][0]

# ✅ Set result label, color & confidence
if prediction >= 0.5:
    label = "PNEUMONIA DETECTED 😷"
    confidence = prediction * 100
    color = "red"
    advice = "⚠️ Seek medical advice immediately. Early treatment helps recovery."
else:
    label = "NORMAL LUNGS 🫁"
    confidence = (1 - prediction) * 100
    color = "green"
    advice = "✅ Your lungs look healthy! Maintain good hygiene and stay fit."

# ✅ Create figure layout
fig = plt.figure(figsize=(7, 8))
grid = plt.GridSpec(3, 1, height_ratios=[4, 0.5, 1.2], hspace=0.4)

# =======================
# 1️⃣ IMAGE DISPLAY
# =======================
ax1 = fig.add_subplot(grid[0])
ax1.imshow(image.load_img(img_path))
ax1.axis("off")

# Diagnosis banner
rect = patches.FancyBboxPatch(
    (0.02, 0.90), 0.96, 0.09, boxstyle="round,pad=0.03",
    transform=ax1.transAxes, facecolor=color, alpha=0.3
)
ax1.add_patch(rect)
ax1.text(0.5, 0.94, label, color=color, fontsize=16, fontweight='bold',
         ha='center', va='center', transform=ax1.transAxes)

# =======================
# 2️⃣ CONFIDENCE BAR
# =======================
ax2 = fig.add_subplot(grid[1])
ax2.barh(["Confidence"], [confidence], color=color, alpha=0.8)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Confidence (%)")
ax2.set_title("AI Confidence Level", fontsize=12)
for i, v in enumerate([confidence]):
    ax2.text(v + 1, i, f"{v:.2f}%", color='black', va='center', fontsize=12)

# =======================
# 3️⃣ SUMMARY BOX
# =======================
ax3 = fig.add_subplot(grid[2])
ax3.axis("off")
ax3.text(0.5, 0.7, f"🧠 AI Diagnosis: {label}",
         ha='center', fontsize=13, fontweight='bold', color=color)
ax3.text(0.5, 0.45, f"📈 Confidence: {confidence:.2f}%",
         ha='center', fontsize=11, color='black')
ax3.text(0.5, 0.2, advice,
         ha='center', fontsize=10, color='darkblue', wrap=True)

plt.suptitle("Pneumonia Detection Result Dashboard", fontsize=15, fontweight='bold')
plt.show()

# ✅ Terminal summary
print("\n🩺 ====== AI DIAGNOSIS SUMMARY ======")
print(f"📂 Image Path: {img_path}")
print(f"🧠 Result: {label}")
print(f"📊 Confidence: {confidence:.2f}%")
print(f"💡 Advice: {advice}")
print("=====================================")
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('pneumonia_model.keras')

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to check for Pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Open image and display it
    img = Image.open(uploaded_file).convert('RGB')  # Convert to 3 channels
    st.image(img, caption='Uploaded Image',use_container_width=True)
    
    # Preprocess image for model
    img = img.resize((150,150))                      # Resize to model input size
    img_array = np.array(img)/255.0                  # Normalize
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Display result
    if prediction > 0.5:
        st.error(f"Pneumonia Detected! Confidence: {prediction*100:.2f}%")
    else:
        st.success(f"Normal Lung! Confidence: {(1-prediction)*100:.2f}%")
