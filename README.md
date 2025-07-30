# Face Recognition Incremental Learning Experiments
FACE RECOGNITION-BASED PORTABLE ATTENDANCE SYSTEM WITH FAST RELEARNING FOR NEW AUDIENCE

## Overview
This experiment implements advanced face recognition with various incremental learning strategies, focusing on Learning without Forgetting (LwF) and joint training approaches.
All experiments are implemented and executed in Google Colab notebooks to ensure reproducibility and accessibility. The corresponding `.ipynb` files can be directly imported into Google Colab for replication of results.

## Model Architectures
- **MobileFaceNet**: Standard model (~2MB, high accuracy)
- **EdgeFace**: Lightweight model (<1MB, optimized for edge/mobile)
- **VGGFace2**: Research-oriented model with frozen backbone (~107MB, high accuracy less suitable for edge/mobile)

## Learning Strategies
1. **Transfer Learning Only**
   - Base approach for new class learning
   - No forgetting mitigation

2. **Joint Training**
   - Mixes old and new class data
   - Reduced forgetting through rehearsal

3. **Learning without Forgetting (LwF)**
   - Knowledge distillation from old model
   - No storage of previous data
   - Configurable temperature and alpha

4. **Combined Joint Training + LwF**
   - Best overall performance
   - Both rehearsal and distillation
   - Most comprehensive forgetting prevention

## Installation Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install insightface onnxruntime
pip install opencv-python scikit-learn matplotlib pandas
pip install Pillow numpy

# Optional (for VGGFace2)
pip install facenet-pytorch
```

## Usage Examples

1. Basic Experiment:
```bash
python unified_face_recognition.py --dataset_path /path/to/dataset --model mobilefacenet --strategy transfer_joint_lwf
```

2. LwF Parameters:
```bash
python unified_face_recognition.py --dataset_path /path/to/dataset --model edgeface --strategy transfer_lwf_only --lwf_weight 0.8 --temperature 4.0
```

## Datasets
The experiments has been evaluated using two standard face recognition datasets:

1. **Labelled Faces in the Wild (LFW)**
   - Used for training and testing in uncontrolled environments
   - Natural variations in pose, lighting, and expression
   - Real-world face recognition scenarios

2. **CUHK03**
   - Used for benchmarking in semi-controlled conditions
   - Captured in campus environment
   - Multiple viewpoints and lighting conditions

## Dataset Structure
```
dataset/
├── person1/
│   ├── image1.jpg
│   └── image2.jpg
│   └── imageN.jpg
├── person2/
└── personN/
```

## Key Features
- Multiple backbone architectures
- Various learning strategies
- Comprehensive evaluation metrics
- Flexible deployment options
- Detailed performance analysis
- Model checkpointing

## Research Applications
- Continual learning research
- Educational demonstrations
- Production systems
- Baseline benchmarking

## Results & Analysis
The experiments generate comprehensive evaluation metrics and analysis:
- Accuracy metrics
- Confusion matrices
- Learning curves
- ROC and PR curves
- Forgetting analysis
- Performance comparisons
- Incremental Learning Analysis

## More Information
For detailed implementation specifics, refer to the experiment notebooks and Python files in the repository.
Each `.ipynb` file in the root directory corresponds to a specific experiment configuration and can be opened directly in Google Colab.
The `py_Files` directory contains the Python script versions of these notebooks for local execution if preferred.
