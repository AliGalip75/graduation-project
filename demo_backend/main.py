import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import glob

# Kök dizini (graduation-project) path'e ekliyoruz ki load_model.py içe aktarılabilsin.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from load_model import load_model_from_opts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seçilebilecek modellerin yolları
MODEL_CONFIGS = {
    "resnet_baseline": {
        "opts": os.path.join(ROOT_DIR, "model", "triplet_baseline", "opts.yaml"),
        "ckpt": os.path.join(ROOT_DIR, "model", "triplet_baseline", "net_69.pth"),
        "return_feature": True
    },
    "resnet_finetuned": {
        "opts": os.path.join(ROOT_DIR, "model", "triplet_finetune", "opts.yaml"),
        "ckpt": os.path.join(ROOT_DIR, "model", "triplet_finetune", "net_19.pth"),
        "return_feature": True
    },
    "swin_baseline": {
        "opts": os.path.join(ROOT_DIR, "model", "swin_triplet_baseline", "opts.yaml"),
        "ckpt": os.path.join(ROOT_DIR, "model", "swin_triplet_baseline", "net_69.pth"),
        "return_feature": True
    },
    "swin_finetuned": {
        "opts": os.path.join(ROOT_DIR, "model", "swin_triplet_finetune", "opts.yaml"),
        "ckpt": os.path.join(ROOT_DIR, "model", "swin_triplet_finetune", "net_19.pth"),
        "return_feature": True
    }
}

loaded_models = {}
gallery_cache = {} # model_name -> (features, paths)

transform_val = transforms.Compose([
    transforms.Resize(size=(224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(model_name: str):
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    cfg = MODEL_CONFIGS.get(model_name)
    if not cfg:
        raise ValueError(f"Bilinmeyen model: {model_name}")
        
    model = load_model_from_opts(cfg["opts"], ckpt=cfg["ckpt"], return_feature=cfg["return_feature"])
    model = model.to(device)
    model.eval()
    loaded_models[model_name] = model
    return model

def extract_feature(model, img_path_or_file):
    if isinstance(img_path_or_file, str):
        img = Image.open(img_path_or_file).convert('RGB')
    else:
        img = Image.open(img_path_or_file.file).convert('RGB')
        
    img = transform_val(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        # Bazı modeller tuple veya list döner (logits, features), bazıları doğrudan features.
        if isinstance(outputs, (tuple, list)):
            ff = outputs[1]
        else:
            ff = outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
    return ff.cpu().numpy()

def load_distractor_gallery(model_name: str, model):
    if model_name in gallery_cache:
        return gallery_cache[model_name]
        
    # Kafa karıştırıcı (distractor) olarak rastgele 1000 resmi kullanıyoruz (daha zorlu bir havuz)
    gallery_dir = os.path.join(ROOT_DIR, "datasets", "VeRi", "image_test")
    all_paths = sorted(glob.glob(os.path.join(gallery_dir, "*.jpg")))
    
    import random
    random.seed(42) # Her seferinde aynı 1000'li grubu seçmesi için sabit seed
    img_paths = random.sample(all_paths, min(1000, len(all_paths)))
    
    features = []
    for path in img_paths:
        feat = extract_feature(model, path)
        features.append(feat)
        
    features = np.concatenate(features, axis=0)
    gallery_cache[model_name] = (features, img_paths)
    return gallery_cache[model_name]

@app.post("/api/search/")
async def search(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        model = get_model(model_name)
    except Exception as e:
        return {"error": str(e)}
        
    # Yüklenen sorgu (query) resminin özelliklerini çıkar
    query_feat = extract_feature(model, file)
    
    # 1. Distractor (Kafa karıştırıcı) galeriyi getir
    dist_features, dist_paths = load_distractor_gallery(model_name, model)
    
    # 2. Yüklenen resmin ID'sini bul ve o araca ait galerideki resimleri (varsa) dinamik olarak havuza dahil et
    filename = file.filename
    query_id = filename.split("_")[0] if "_" in filename else None
    
    dynamic_features = []
    dynamic_paths = []
    
    if query_id and query_id.isdigit():
        gallery_dir = os.path.join(ROOT_DIR, "datasets", "VeRi", "image_test")
        target_paths = glob.glob(os.path.join(gallery_dir, f"{query_id}_*.jpg"))
        
        for path in target_paths:
            # Sadece distractor içinde zaten yoksa işle (tekrarı önlemek için)
            if path not in dist_paths:
                feat = extract_feature(model, path)
                dynamic_features.append(feat)
                dynamic_paths.append(path)
                
    if dynamic_features:
        dynamic_features_arr = np.concatenate(dynamic_features, axis=0)
        final_gallery_features = np.concatenate([dist_features, dynamic_features_arr], axis=0)
        final_gallery_paths = dist_paths + dynamic_paths
    else:
        final_gallery_features = dist_features
        final_gallery_paths = dist_paths
    
    # Kosinüs Benzerliği (Cosine Similarity) hesapla
    scores = np.dot(final_gallery_features, query_feat.T).flatten()
    
    # En yüksek skora sahip 12 resmi bul
    top_indices = np.argsort(scores)[::-1][:12]
    
    results = []
    for idx in top_indices:
        results.append({
            "path": os.path.basename(final_gallery_paths[idx]),
            "score": round(float(scores[idx]) * 100, 1) # Yüzdelik dilime çevir
        })
        
    return {"results": results}

@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Galeri resimlerini frontend'e sunmak için yardımcı endpoint"""
    path = os.path.join(ROOT_DIR, "datasets", "VeRi", "image_test", filename)
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Dosya bulunamadı"}
