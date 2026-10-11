# Image Captioning with Transformer Decoder & Attention Visualization

This project implements an end-to-end Image Captioning pipeline built with **TensorFlow / Keras**. It combines a pre-trained **MobileNetV2** (acting as a visual feature extractor) with a custom **Transformer Decoder** architecture utilizing causal self-attention, cross-attention mechanisms, and attention weight visualization.

## 🚀 Key Features
* **Hybrid Architecture:** Uses a frozen CNN backbone (`MobileNetV2`) to extract spatial image features and a custom Transformer-based text decoder.
* **Custom Layers Implementation:** Built from scratch using modern Keras functional/subclassing APIs:
  * `SeqEmbedding` (Token & Positional Embeddings)
  * `CausalSelfAttention` & `CrossAttention` (Multi-Head Attention blocks with Layer Normalization)
  * `FeedForward` and `DecoderLayer` blocks
* **Attention Map Visualization:** Includes tools to process, resize, and overlay attention weights onto input images to visualize which regions of the image the model focuses on when generating specific words.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras**
* **Einops** (tensor rearrangements)
* **Matplotlib & PIL** (image processing and visualization)

## 📂 Project Structure
* `model.py` (or main script) — Core architecture definitions (`Captioner` model, attention layers, token management) and visualization pipeline.
* `Herons.jpg` — Sample input image for testing caption generation and attention plotting.

## 📊 How It Works
1. **Feature Extraction:** An input image $(224 \times 224 \times 3)$ is passed through MobileNetV2 to extract feature maps, which are then rearranged using `einops` into sequential vectors.
2. **Decoding & Attention:** The text sequence is embedded via positional and token embeddings, passing through multiple Transformer decoder layers where **Cross-Attention** links textual tokens with image regions.
3. **Visualization:** The script loads an image, generates tokens, computes attention score matrices, and plots heatmaps showing visual focus areas for each generated word.
