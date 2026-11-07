# Deep Learning & AI Engineering Learning Roadmap

A comprehensive, hands-on repository tracking my journey through core deep learning principles, modern architectures, LLM engineering, MLOps, low-level CUDA programming, and robotics.

---

## 📌 Curriculum Overview

This repository is structured into modular topics covering foundational mathematics, custom PyTorch implementations, modern generative/3D models, AI engineering frameworks, and hardware integration.

### 1. 📐 Mathematical Foundations
- **Linear Algebra**: Vectors, Matrices, Eigenvalues/Eigenvectors, SVD, Tensor operations
- **Calculus**: Multivariable calculus, Partial derivatives, Jacobians, Hessians, Chain rule
- **Probability & Statistics**: Bayes' theorem, Probability distributions, Information theory (Entropy, KL Divergence, Cross-Entropy)
- **Optimization**: Convex optimization, Gradient-based methods, Constrained optimization

### 2. 🧠 Deep Learning Core & Autograd
- **Building Blocks**: Perceptrons, Artificial Neurons, Activation Functions (ReLU, GELU, Swish, SiLU)
- **Mechanics**: Automatic Differentiation (Autograd from scratch), Forward & Backward passes
- **Optimization & Loss**: Cross-Entropy, MSE, Focal Loss, SGD, Momentum, Adam, AdamW
- **Training Techniques**: Backpropagation, Gradient Clipping, Normalization (LayerNorm, RMSNorm, BatchNorm)

### 3. 🔥 PyTorch & Architecture Building
- **PyTorch Fundamentals**: Tensors, Custom Datasets, DataLoaders, Modules, Hooks
- **Vision Architectures**: ConvNeXt, ResNet, Vision Transformers (ViT)
- **State Space Models (SSMs)**: Selective State Spaces, Mamba architecture implementations

### 4. 👁️ Computer Vision & Spatial AI
- **Convolutional Networks**: CNN design patterns, Spatial hierarchies, Feature extraction
- **Segmentation & Detection**: U-Net, Mask R-CNN, Semantic & Instance segmentation
- **Vision-Language**: CLIP, SigLIP, Visual Question Answering (VQA)

### 5. 🎨 Advanced Generative & 3D Models
- **Diffusion Models**: Denoising Diffusion Probabilistic Models (DDPM), Latent Diffusion
- **Diffusion Transformers**: DiT (Diffusion Transformer) implementation & fine-tuning
- **Generative Vision**: SDXL VAE & Latent Space Decoders
- **3D Representations**: 3D Gaussian Splatting (3DGS), Point Clouds, TSDF Volume Fusion, NeRFs

### 6. ⚡ Transformers Deep Dive
- **Core Components**: Scaled Dot-Product Attention, Self-Attention, Cross-Attention
- **Tokenization**: BPE, WordPiece, SentencePiece, Tiktoken
- **Positional Encodings**: Absolute, Relative, Rotary Position Embeddings (RoPE)

### 7. 🚀 LLMs from Scratch
- **Architecture**: Decoder-only Transformer pipeline (Llama/Qwen style)
- **Attention Variants**: Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA)
- **Training Pipeline**: Pre-training loop, Distributed Data Parallel (DDP), FSDP

### 8. 🛠️ Applied LLM & Agentic Systems
- **Vector Systems**: Vector DBs, Embedding spaces, Similarity search algorithms (HNSW, Flat)
- **RAG Architectures**: Naive RAG, Advanced RAG (HyDE, Re-ranking, Parent-Child indexing)
- **Memory & Context**: Ephemeral/Long-term chat memory, Graph-based memory
- **Agent Orchestration**: Tool calling, ReAct pattern, Multi-agent collaboration

### 9. 🎛️ Post-Training & Fine-Tuning
- **Supervised Fine-Tuning (SFT)**: Dataset preparation, Instruction formatting
- **Parameter-Efficient Fine-Tuning (PEFT)**: LoRA, QLoRA, Prefix Tuning, Adapters
- **Alignment**: RLHF, Direct Preference Optimization (DPO), KTO
- **Quantization**: Post-Training Quantization (PTQ), AWQ, GGUF, GPTQ, EXL2

### 10. 👁️🤖 Embodied AI & Multimodal Agents
- **Multimodal Models**: Vision-Language-Action (VLA) models
- **Robotic Decision Making**: Perceptual grounding, Task planning with VLMs
- **Interactive Agents**: Real-time sensory processing & tool integration

### 11. 🧱 AI Engineering & Frameworks
- **Java/Kotlin Enterprise AI**: Spring AI integration, Vector Stores, JDBC Chat Memory
- **Graph Orchestration**: LangGraph stateful multi-agent workflows
- **Emerging Frameworks**: Embable and custom agentic frameworks

### 12. ☁️ MLOps & High-Performance Serving
- **Serving Engines**: vLLM, Triton Inference Server, Ollama, TGI
- **Model Formats & Compression**: GGUF, AWQ, ONNX conversions
- **Infrastructure & Cloud**: Distributed training, Model deployment strategies, AWS AI/ML ecosystem

### 13. 🤖 AI Automation & Workflow Pipelines
- **Agent Frameworks**: CrewAI, Hermes Agent Engine
- **Workflow Automation**: n8n integration, Webhooks, Pipeline automation
- **Generative Media Automation**: ComfyUI API automation & custom node creation

### 14. ⚡ Low-Level AI Development & CUDA
- **GPU Architecture**: Streaming Multiprocessors, CUDA Cores, Tensor Cores, Memory Hierarchy
- **Kernel Programming**: Writing custom CUDA kernels, PyTorch C++ extensions
- **Performance Optimization**: Memory coalescing, Shared memory usage, Profiling (PyTorch Profiler, Nsight Systems)

### 15. 🤖 Robotics & Mechatronics
- **ROS 2**: Nodes, Topics, Services, Actions, Custom Interfaces, URDF modeling
- **Sensors & Perception**: Camera integration, OpenCV processing, LiDAR data pipelines
- **Mechatronics & Control**: Microcontroller communications (Arduino/ESP32 serial protocols), Motor drivers, Kinematics

---

## 🛠️ Tech Stack & Tooling

- **Languages**: Python 3.11+, C++, CUDA, Java, Kotlin
- **DL / ML Frameworks**: PyTorch, torchvision, torchaudio, Hugging Face (Transformers, PEFT, Datasets)
- **Package & Environment Management**: [uv](https://github.com/astral-sh/uv)
- **Serving & Hardware**: CUDA 12.x, vLLM, Triton, Spring AI, ROS 2

---

## 🚀 Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### Prerequisites
Make sure `uv` is installed on your system:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh