# model_loader.py
import os
import zipfile
import gdown

def download_and_extract_zip(file_id, target_folder, zip_name="model.zip"):
    if os.path.exists(os.path.join(target_folder, "model.safetensors")):
        return  # Already downloaded

    os.makedirs(target_folder, exist_ok=True)
    zip_path = os.path.join(target_folder, zip_name)
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"🔽 Downloading model: {target_folder}")
    gdown.download(url, zip_path, quiet=False)

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    os.remove(zip_path)
    print(f"✅ Model ready in {target_folder}")

def ensure_models_downloaded():
    download_and_extract_zip("1B58UQgYZ9B34gmkh03E0P3C74s6CaUFo", "distilbert_resume_classifier_v2")
    download_and_extract_zip("1kctJ9apPMPCi0pHEkipdsQvQkukiRJ4X", "t5model_v5")
