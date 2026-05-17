import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import os
from rembg import remove
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from datetime import datetime

# Muat model
model_warna = tf.keras.models.load_model(r'model/30warnamobilenetv2_model.h5')
model_kebersihan = tf.keras.models.load_model(r'model/50kebersihan_model.h5')
model_keretakan = tf.keras.models.load_model(r'model/newkeretakan_model.h5')

# Label klasifikasi
warna_label = ['Coklat Muda', 'Coklat Pucat', 'Coklat Tua']
kebersihan_label = ['Bersih', 'Kotor', 'Sedikit Kotor']
keretakan_label = ['Retak Parah', 'Retak Ringan', 'Tidak Retak']

# Path gambar
image_path = r'hehe.jpeg'

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path,time, img_size=(128, 128)):
    try:
        timestamp = time

        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        img_no_bg = remove(img_data)

        img_array = np.frombuffer(img_no_bg, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(output_dir, f"removed_bg_{timestamp}.png"), image)

        if image.shape[-1] == 4:
            alpha = image[:, :, 3]
            mask = alpha > 0
            image = image[:, :, :3] * np.expand_dims(mask, axis=-1)
            image = image.astype(np.uint8)

        original = cv2.imread(image_path)
        cv2.imwrite(os.path.join(output_dir, f"original_{timestamp}.png"), original)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        if w == 0 or h == 0:
            raise ValueError("Objek tidak ditemukan setelah menghapus background.")

        cropped = image_rgb[y:y + h, x:x + w]
        #cv2.imwrite(os.path.join(output_dir, f"cropped_{timestamp}.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        blurred = cv2.GaussianBlur(cropped, (5, 5), 1.0)
        sharpened_rgb = cv2.addWeighted(cropped, 1.5, blurred, -0.3, 0)
        cv2.imwrite(os.path.join(output_dir, f"sharpened_{timestamp}.png"), cv2.cvtColor(sharpened_rgb, cv2.COLOR_RGB2BGR))

        resized = cv2.resize(sharpened_rgb, img_size)
        cv2.imwrite(os.path.join(output_dir, f"resized_{timestamp}.png"), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        return np.expand_dims(resized / 255.0, axis=0), sharpened_rgb, image_rgb, resized

    except Exception as e:
        print(f"Error dalam preprocessing: {e}")
        return None, None, None, None

def extract_crack(resized, timestamp, output_dir):
    if resized.dtype != np.uint8:
        resized = (resized * 255).astype(np.uint8)

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(output_dir, f"gray_{timestamp}.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(os.path.join(output_dir, f"canny_{timestamp}.png"), edges)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    crack_highlight = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return crack_highlight

def extract_rgb_features(image):
    return np.mean(image, axis=(0, 1))

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]
    return np.array(features)

def save_features_to_csv(image_path, rgb_features, glcm_features, csv_path='features.csv'):
    features = np.concatenate([rgb_features, glcm_features])
    df = pd.DataFrame([np.append(features, image_path)], 
                      columns=['R', 'G', 'B', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'image_path'])
    
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def predict_quality(image_path,time):
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_array, cropped_img, _, img_resized = preprocess_image(image_path,time)

    if img_array is None:
        print("Gagal memproses gambar.")
        return

    # Ekstraksi fitur
    rgb_feat = extract_rgb_features(cropped_img)
    glcm_feat = extract_glcm_features(cropped_img)
    save_features_to_csv(image_path, rgb_feat, glcm_feat)

    # Prediksi warna & kebersihan
    pred_warna = model_warna.predict(img_array)
    pred_kebersihan = model_kebersihan.predict(img_array)
    warna = warna_label[np.argmax(pred_warna)]
    kebersihan = kebersihan_label[np.argmax(pred_kebersihan)]

    # Prediksi keretakan
    img_resized = cv2.resize(cropped_img, (128, 128)) / 255.0
    input_array = np.expand_dims(img_resized, axis=0)
    pred_keretakan = model_keretakan.predict(input_array)
    keretakan = keretakan_label[np.argmax(pred_keretakan)]
    extract_crack(img_resized, time, output_dir)

    # Klasifikasi mutu berdasarkan parameter
    if keretakan == 'Retak Parah':
        mutu = 'Mutu III'
    elif keretakan == 'Retak Ringan':
        mutu = 'Mutu II'
    elif keretakan == 'Tidak Retak':
        if warna == 'Coklat Tua' and kebersihan in ['Bersih', 'Sedikit Kotor', 'Kotor']:
            mutu = 'Mutu I'
        elif warna == 'Coklat Muda' and kebersihan in ['Bersih', 'Sedikit Kotor', 'Kotor']:
            mutu = 'Mutu II'
        else:
            mutu = 'Mutu III'
    else:
        mutu = 'Tidak terdeteksi'
    # Tampilkan nilai RGB dan GLCM
    print(f"Nilai RGB: R: {rgb_feat[0]:.2f}, G: {rgb_feat[1]:.2f}, B: {rgb_feat[2]:.2f}")
    print(f"GLCM - Contrast: {glcm_feat[0]:.4f}, Dissimilarity: {glcm_feat[1]:.4f}, Homogeneity: {glcm_feat[2]:.4f}, Energy: {glcm_feat[3]:.4f}, Correlation: {glcm_feat[4]:.4f}")
    print(f"Mutu Telur: {mutu}")
    print(f"Warna Telur: {warna}")
    print(f"Kebersihan Telur: {kebersihan}")
    Probabilitas_Warna = round(np.max(pred_warna), 2)
    Probabilitas_Kebersihan = round(np.max(pred_kebersihan), 2)
    Probabilitas_Keretakan = round(np.max(pred_keretakan), 2)
    return mutu, warna, kebersihan, keretakan, \
       np.around(float(Probabilitas_Warna), 2), \
       np.around(float(Probabilitas_Kebersihan), 2), \
       np.around(float(Probabilitas_Keretakan), 2), \
       rgb_feat, glcm_feat
    # return mutu, warna, kebersihan,keretakan, np.around(float(Probabilitas_Warna), 2), np.around(float(Probabilitas_Kebersihan), 2),np.around(float(Probabilitas_Keretakan), 2)
#     plt.imshow(original_img)
#     plt.title(f"Mutu: {mutu}\nWarna: {warna} ({np.max(pred_warna):.2%})\nKebersihan: {kebersihan} ({np.max(pred_kebersihan):.2%})\nKeretakan: {keretakan} ({np.max(pred_keretakan):.2%})")
#     plt.axis('off')
#     plt.show()

# predict_quality(image_path)
