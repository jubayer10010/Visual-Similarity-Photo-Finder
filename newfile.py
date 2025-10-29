import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
import glob

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            embedding = model(batch_t)
            
        return embedding.flatten().numpy()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_gallery_index(gallery_dir):
    print(f"--- 1. Indexing Gallery Photos in '{gallery_dir}' ---")
    
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)
        print("NOTE: Created a dummy gallery folder. Add your test images here!")
        return None, None
        
    image_paths = glob.glob(os.path.join(gallery_dir, '*.[jJpP]*'))
    if not image_paths:
        print("No images found in the gallery folder. Please add some test images.")
        return None, None
        
    gallery_embeddings = []
    file_names = []
    
    for path in image_paths:
        embedding = get_image_embedding(path)
        if embedding is not None:
            gallery_embeddings.append(embedding)
            file_names.append(os.path.basename(path))
            print(f"Indexed: {os.path.basename(path)}")
            
    if not gallery_embeddings:
        return None, None
        
    X = np.array(gallery_embeddings)
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(X)
    
    return nn_model, file_names

def find_similar_images(nn_model, file_names, query_path):
    print(f"\n--- 2. Querying with '{os.path.basename(query_path)}' ---")
    
    query_embedding = get_image_embedding(query_path)
    if query_embedding is None:
        return
        
    query_reshaped = query_embedding.reshape(1, -1)
    
    distances, indices = nn_model.kneighbors(query_reshaped)
    
    print("\n✅ Found the following similar images in your gallery:")
    
    for i, index in enumerate(indices.flatten()):
        similarity = 1 - distances.flatten()[i] 
        print(f"  {i+1}. {file_names[index]} (Similarity: {similarity:.4f})")

if __name__ == "__main__":
    
    GALLERY_FOLDER = "my_photo_gallery"
    example_query_name = "image1.jpg" 
    
    nn_model, file_names = build_gallery_index(GALLERY_FOLDER)
    
    if nn_model:
        QUERY_IMAGE_PATH = os.path.join(GALLERY_FOLDER, example_query_name)
        if os.path.exists(QUERY_IMAGE_PATH):
            find_similar_images(nn_model, file_names, QUERY_IMAGE_PATH)
        else:
            print(f"\n❌ ERROR: Query image not found at {QUERY_IMAGE_PATH}. Please check the filename.")
