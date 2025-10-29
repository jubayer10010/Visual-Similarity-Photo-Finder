# Visual Similarity Photo Finder (Content-Based Image Retrieval)

## 💡 The Problem

As photo galleries grow larger, finding a specific picture based on visual content becomes nearly impossible. Furthermore, when using social media, there is currently no efficient way to find a similar image from your private gallery to share in a comment, forcing users to scroll endlessly.

## ✨ The Solution

This project introduces a **Content-Based Image Retrieval (CBIR)** system designed to solve the "lost photo" problem and enable "visual commenting."

### Key Features:

1.  **Gallery Indexing:** Converts all gallery images into high-dimensional numerical **embeddings** (unique mathematical fingerprints) using a pre-trained **ResNet-50** deep learning model.
2.  **Query-by-Image:** Allows a user to input a "query image" (e.g., a social media photo).
3.  **Instant Retrieval:** Uses **Nearest Neighbor** search to quickly find and rank the most visually similar images from the user's gallery.

## ⚙️ Technical Core

The core functionality is implemented in `image_search_core.py`, which leverages **Transfer Learning** (PyTorch/torchvision) and **Vector Search** (scikit-learn).

## 🚀 How to Run the Prototype

1.  Clone this repository.
2.  Install dependencies: `pip install torch torchvision numpy scikit-learn pillow`
3.  Create a folder named `my_photo_gallery` in the project root.
4.  Add your test images to `my_photo_gallery`.
5.  Edit the `example_query_name` variable in `image_search_core.py` to match one of your test images.
6.  Run the script: `python image_search_core.py`
