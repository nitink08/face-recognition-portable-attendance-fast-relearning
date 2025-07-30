# -*- coding: utf-8 -*-
"""
Unified Face Recognition Incremental Learning Framework
A comprehensive framework supporting multiple architectures and learning strategies:
- Models: MobileFaceNet, EdgeFace, VGGFace2
- Strategies: Transfer Only, Joint Training, LwF, Combined approaches
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import cv2
import time
import pickle
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import warnings
import copy
import argparse
from enum import Enum
import sys
import itertools

# Check and import optional dependencies
VGGFACE2_AVAILABLE = False
try:
    from facenet_pytorch import InceptionResnetV1
    VGGFACE2_AVAILABLE = True
except ImportError:
    pass

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ModelType(Enum):
    """Supported model architectures"""
    MOBILEFACENET = "mobilefacenet"
    EDGEFACE = "edgeface"
    VGGFACE2 = "vggface2"

class LearningStrategy(Enum):
    """Supported learning strategies"""
    TRANSFER_ONLY = "transfer_only"
    TRANSFER_JOINT = "transfer_joint"
    TRANSFER_JOINT_LWF = "transfer_joint_lwf"
    TRANSFER_LWF_ONLY = "transfer_lwf_only"

# Model configuration mapping
MODEL_CONFIGS = {
    ModelType.MOBILEFACENET: {
        'available': True,
        'feature_dim': 512,
        'input_size': (640, 640),
        'det_size': (640, 640),
        'insightface_model': 'buffalo_l',
        'classifier_hidden': [256],
        'description': 'Standard face recognition model with full InsightFace features'
    },
    ModelType.EDGEFACE: {
        'available': True,
        'feature_dim': 512,
        'input_size': (112, 112),
        'det_size': (112, 112),
        'insightface_model': 'buffalo_sc',
        'classifier_hidden': [128, 64],
        'description': 'Lightweight model optimized for edge computing'
    },
    ModelType.VGGFACE2: {
        'available': VGGFACE2_AVAILABLE,
        'feature_dim': 512,
        'input_size': (160, 160),
        'det_size': None,  # Uses PyTorch transforms
        'insightface_model': None,  # Uses facenet-pytorch
        'classifier_hidden': [256],
        'description': 'Pre-trained VGGFace2 model with frozen backbone'
    }
}

class FaceDataset(Dataset):
    """Custom dataset for face recognition training"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss for Learning without Forgetting"""
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        """Compute combined loss for new task learning and knowledge preservation"""
        classification_loss = self.ce_loss(student_logits, labels)

        if teacher_logits is not None and teacher_logits.size(1) > 0:
            num_old_classes = teacher_logits.size(1)
            student_old_logits = student_logits[:, :num_old_classes]

            soft_student = F.log_softmax(student_old_logits / self.temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)

            distillation_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
            total_loss = (1 - self.alpha) * classification_loss + self.alpha * distillation_loss

            return total_loss, classification_loss, distillation_loss
        else:
            return classification_loss, classification_loss, torch.tensor(0.0)

class UnifiedFaceModel(nn.Module):
    """Unified model supporting all three architectures with optional LwF"""
    def __init__(self, num_classes, model_type=ModelType.MOBILEFACENET, 
                 feature_dim=512, temperature=3.0, lwf_weight=1.0):
        super(UnifiedFaceModel, self).__init__()

        # Validate model availability
        if not MODEL_CONFIGS[model_type]['available']:
            available_models = [m.value for m in ModelType if MODEL_CONFIGS[m]['available']]
            raise ValueError(f"Model {model_type.value} not available. Available models: {available_models}")

        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.feature_dim = self.config['feature_dim']
        self.temperature = temperature
        self.lwf_weight = lwf_weight
        self.num_old_classes = 0
        self.is_first_task = True

        # Initialize feature extraction
        self._initialize_feature_extractor()
        
        # Initialize classifier
        self.classifier = self._build_classifier(num_classes)
        self._initialize_classifier()
        
        # LwF components
        self.old_model = None
        self.old_feature_extractor = None

    def _initialize_feature_extractor(self):
        """Initialize feature extractor based on model type"""
        if self.model_type == ModelType.VGGFACE2:
            print("Loading pre-trained VGGFace2 backbone...")
            self.backbone = InceptionResnetV1(
                pretrained='vggface2',
                classify=False,
                num_classes=None
            )
            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Frozen {sum(1 for p in self.backbone.parameters())} backbone parameters")
        else:
            # Initialize InsightFace app for MobileFaceNet and EdgeFace
            self.app = self._initialize_insightface()

    def _initialize_insightface(self):
        """Initialize InsightFace app with fallback options"""
        model_configs = [
            {'name': self.config['insightface_model'], 'det_size': self.config['det_size']},
            {'name': 'buffalo_l', 'det_size': (640, 640)},
            {'name': 'buffalo_m', 'det_size': (640, 640)},
            {'name': 'buffalo_s', 'det_size': (320, 320)},
        ]

        for config in model_configs:
            try:
                app = FaceAnalysis(name=config['name'], providers=['CPUExecutionProvider'])
                app.prepare(ctx_id=0, det_size=config['det_size'])
                
                # Test the model
                test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                _ = app.get(test_img)
                print(f"Successfully initialized {config['name']} for {self.model_type.value}")
                return app
            except Exception as e:
                print(f"Failed to initialize {config['name']}: {e}")
                continue

        raise RuntimeError(f"Could not initialize any InsightFace model for {self.model_type.value}")

    def _build_classifier(self, num_classes):
        """Build classifier based on model configuration"""
        layers = []
        input_dim = self.feature_dim
        
        # Add hidden layers
        for hidden_dim in self.config['classifier_hidden']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5 if hidden_dim > 128 else 0.3)
            ])
            input_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        return nn.Sequential(*layers)

    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_features=False):
        """Forward pass with optional feature return for LwF"""
        if self.model_type == ModelType.VGGFACE2:
            with torch.no_grad():
                features = self.backbone(x)
            logits = self.classifier(features)
            if return_features:
                intermediate_features = self.classifier[:-1](features)
                return logits, intermediate_features
        else:
            logits = self.classifier(x)
            features = self.classifier[:-1](x) if return_features else None

        if return_features:
            return logits, features
        return logits

    def extract_features(self, image_path):
        """Extract features using the appropriate model"""
        try:
            if self.model_type == ModelType.VGGFACE2:
                return self._extract_vggface2_features(image_path)
            else:
                return self._extract_insightface_features(image_path)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(self.feature_dim)

    def _extract_vggface2_features(self, image_path):
        """Extract features using VGGFace2"""
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(self.feature_dim)
        
        # Convert BGR to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.config['input_size'])
        
        # Convert to PIL and apply transforms
        img_pil = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img_pil).unsqueeze(0)
        
        # Extract features using frozen backbone
        with torch.no_grad():
            features = self.backbone(img_tensor)
            return features.squeeze().numpy()

    def _extract_insightface_features(self, image_path):
        """Extract features using InsightFace"""
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(self.feature_dim)

        # Resize based on model type
        if self.model_type == ModelType.EDGEFACE:
            img = cv2.resize(img, self.config['input_size'])
        
        faces = self.app.get(img)

        if len(faces) > 0:
            face = max(faces, key=lambda x: x.det_score if hasattr(x, 'det_score') else x.bbox[2] * x.bbox[3])
            embedding = face.embedding

            # Handle dimension mismatch
            if len(embedding) != self.feature_dim:
                if len(embedding) < self.feature_dim:
                    padded_embedding = np.zeros(self.feature_dim)
                    padded_embedding[:len(embedding)] = embedding
                    embedding = padded_embedding
                else:
                    embedding = embedding[:self.feature_dim]

            # L2 normalization
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        else:
            return np.zeros(self.feature_dim)

    def save_old_model(self):
        """Save current model as old model for LwF"""
        if not self.is_first_task:
            self.old_model = copy.deepcopy(self.classifier)
            for param in self.old_model.parameters():
                param.requires_grad = False
            self.old_model.eval()

            # For models with multiple hidden layers, save feature extractor
            if len(self.config['classifier_hidden']) > 1:
                self.old_feature_extractor = copy.deepcopy(self.classifier[:-1])
                for param in self.old_feature_extractor.parameters():
                    param.requires_grad = False
                self.old_feature_extractor.eval()

    def update_classifier(self, num_new_total_classes):
        """Update classifier for new number of classes"""
        self.num_old_classes = self.classifier[-1].out_features if not self.is_first_task else 0

        if not self.is_first_task:
            self.save_old_model()

        old_weight = self.classifier[-1].weight.data
        old_bias = self.classifier[-1].bias.data

        # Get input dimension for final layer
        final_input_dim = self.config['classifier_hidden'][-1]
        new_final_layer = nn.Linear(final_input_dim, num_new_total_classes)

        # Initialize new layer
        nn.init.xavier_uniform_(new_final_layer.weight)
        nn.init.constant_(new_final_layer.bias, 0)

        # Copy old weights for existing classes
        if not self.is_first_task and num_new_total_classes >= old_weight.size(0):
            new_final_layer.weight.data[:old_weight.size(0)] = old_weight
            new_final_layer.bias.data[:old_bias.size(0)] = old_bias

        self.classifier[-1] = new_final_layer
        self.is_first_task = False

    def compute_lwf_loss(self, current_logits, raw_features, temperature=None):
        """Compute Learning without Forgetting loss"""
        if self.old_model is None or self.is_first_task or self.num_old_classes == 0:
            return torch.tensor(0.0, device=current_logits.device)

        if temperature is None:
            temperature = self.temperature

        try:
            with torch.no_grad():
                if hasattr(self, 'old_feature_extractor') and self.old_feature_extractor is not None:
                    old_processed_features = self.old_feature_extractor(raw_features)
                    old_logits = self.old_model[-1](old_processed_features)
                else:
                    old_logits = self.old_model(raw_features)

            if self.num_old_classes > 0 and current_logits.size(1) >= self.num_old_classes:
                current_old_logits = current_logits[:, :self.num_old_classes]

                if old_logits.size(1) != current_old_logits.size(1):
                    min_classes = min(old_logits.size(1), current_old_logits.size(1))
                    old_logits = old_logits[:, :min_classes]
                    current_old_logits = current_old_logits[:, :min_classes]

                    if min_classes == 0:
                        return torch.tensor(0.0, device=current_logits.device)

                soft_targets = F.softmax(old_logits / temperature, dim=1)
                soft_predictions = F.log_softmax(current_old_logits / temperature, dim=1)

                lwf_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
                lwf_loss *= (temperature ** 2)

                return lwf_loss
            else:
                return torch.tensor(0.0, device=current_logits.device)

        except Exception as e:
            print(f"Error in LwF loss computation: {e}")
            return torch.tensor(0.0, device=current_logits.device)

class UnifiedFaceRecognitionResearcher:
    """Main research class for unified face recognition experiments"""
    def __init__(self, dataset_path, model_type=ModelType.MOBILEFACENET, 
                 learning_strategy=LearningStrategy.TRANSFER_ONLY,
                 batch_size=5, lwf_weight=0.5, temperature=3.0):
        
        # Validate model availability
        if not MODEL_CONFIGS[model_type]['available']:
            available_models = [m.value for m in ModelType if MODEL_CONFIGS[m]['available']]
            raise ValueError(f"Model {model_type.value} not available. Available models: {available_models}")
        
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.learning_strategy = learning_strategy
        self.batch_size = batch_size
        self.lwf_weight = lwf_weight
        self.temperature = temperature
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Components based on strategy
        if learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
            self.kd_loss = KnowledgeDistillationLoss(temperature=temperature, alpha=lwf_weight)
        
        # Data organization
        self.person_folders = []
        self.trained_persons = []
        self.test_data = {}
        self.training_history = []
        
        # Performance tracking
        self.metrics_history = []
        self.lwf_losses = []
        self.feature_cache = {}
        
        print(f"Initialized Unified Face Recognition Framework")
        print(f"Model Type: {model_type.value}")
        print(f"Model Description: {MODEL_CONFIGS[model_type]['description']}")
        print(f"Learning Strategy: {learning_strategy.value}")
        print(f"Device: {self.device}")
        if learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
            print(f"LwF Configuration: Weight={lwf_weight}, Temperature={temperature}")

    def organize_dataset(self):
        """Organize dataset into person folders"""
        self.person_folders = [f for f in os.listdir(self.dataset_path)
                              if os.path.isdir(os.path.join(self.dataset_path, f))]
        self.person_folders.sort()
        print(f"Found {len(self.person_folders)} person folders")

        for person in self.person_folders:
            person_path = os.path.join(self.dataset_path, person)
            images = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            if len(images) > 1:
                random.shuffle(images)
                test_img = images[0]
                train_imgs = images[1:]
                self.test_data[person] = {
                    'test_image': os.path.join(person_path, test_img),
                    'train_images': [os.path.join(person_path, img) for img in train_imgs]
                }
            elif len(images) == 1:
                self.test_data[person] = {
                    'test_image': os.path.join(person_path, images[0]),
                    'train_images': [os.path.join(person_path, images[0])]
                }

    def extract_features_batch(self, image_paths, use_cache=True):
        """Extract features for a batch of images"""
        features = []
        config = MODEL_CONFIGS[self.model_type]
        feature_dim = config['feature_dim']

        successful_extractions = 0
        for idx, img_path in enumerate(image_paths):
            if use_cache and img_path in self.feature_cache:
                features.append(self.feature_cache[img_path])
                successful_extractions += 1
                continue

            feature = self.model.extract_features(img_path)
            features.append(feature)

            if np.any(feature):
                successful_extractions += 1
                if use_cache:
                    self.feature_cache[img_path] = feature

        print(f"Successfully extracted features from {successful_extractions}/{len(image_paths)} images")
        return np.array(features)

    def prepare_training_data(self, current_batch_persons, previous_persons, augment_factor=2):
        """Prepare training data based on learning strategy"""
        current_images = []
        current_labels = []

        # Current batch data
        for i, person in enumerate(current_batch_persons):
            if person in self.test_data:
                person_images = self.test_data[person]['train_images']
                current_images.extend(person_images)
                current_labels.extend([len(previous_persons) + i] * len(person_images))

        # Handle previous data based on strategy
        if self.learning_strategy == LearningStrategy.TRANSFER_ONLY:
            return current_images, current_labels
        elif self.learning_strategy == LearningStrategy.TRANSFER_LWF_ONLY:
            return current_images, current_labels
        else:
            # Joint training strategies
            previous_images = []
            previous_labels = []

            if previous_persons:
                target_samples = max(1, len(current_images) // len(previous_persons))
                for i, person in enumerate(previous_persons):
                    if person in self.test_data:
                        person_train_images = self.test_data[person]['train_images']
                        if len(person_train_images) <= target_samples:
                            sampled_images = person_train_images * augment_factor
                        else:
                            sampled_images = random.sample(person_train_images, target_samples) * augment_factor
                        
                        previous_images.extend(sampled_images)
                        previous_labels.extend([i] * len(sampled_images))

            all_images = current_images + previous_images
            all_labels = current_labels + previous_labels
            return all_images, all_labels

    def train_batch(self, batch_persons, is_first_batch=True):
        """Train model on a batch of persons"""
        print(f"\nTraining batch using {self.learning_strategy.value}: {batch_persons}")
        start_time = time.time()

        # Prepare training data
        if is_first_batch:
            train_images, train_labels = self.prepare_training_data(batch_persons, [], augment_factor=1)
            num_classes = len(batch_persons)
        else:
            train_images, train_labels = self.prepare_training_data(batch_persons, self.trained_persons, augment_factor=2)
            num_classes = len(self.trained_persons) + len(batch_persons)

        if not train_images:
            print("Warning: No training images found")
            return 0

        # Extract features
        features = self.extract_features_batch(train_images, use_cache=True)
        actual_feature_dim = features.shape[1] if len(features) > 0 else MODEL_CONFIGS[self.model_type]['feature_dim']

        # Initialize or update model
        if is_first_batch:
            self.model = UnifiedFaceModel(
                num_classes, 
                model_type=self.model_type,
                feature_dim=actual_feature_dim,
                temperature=self.temperature,
                lwf_weight=self.lwf_weight
            )
        else:
            self.model.update_classifier(num_classes)

        # Filter valid features
        valid_indices = [i for i, feat in enumerate(features) if np.any(feat)]
        if len(valid_indices) < len(features):
            print(f"Warning: {len(features) - len(valid_indices)} feature extractions failed")
            features = features[valid_indices]
            train_labels = [train_labels[i] for i in valid_indices]

        if len(features) == 0:
            print("Error: No valid features extracted")
            return 0

        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        labels_tensor = torch.LongTensor(train_labels).to(self.device)
        self.model.to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(self.model.classifier.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=min(32, len(features)), shuffle=True)

        # Training loop
        self.model.train()
        epoch_lwf_losses = []

        for epoch in range(25):
            total_loss = 0
            total_ce_loss = 0
            total_lwf_loss = 0
            correct = 0
            total = 0

            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                if self.learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
                    outputs, processed_features = self.model(batch_features, return_features=True)
                    ce_loss = criterion(outputs, batch_labels)
                    
                    # LwF loss
                    lwf_loss = self.model.compute_lwf_loss(outputs, batch_features, self.temperature)
                    total_batch_loss = ce_loss + self.lwf_weight * lwf_loss
                    
                    total_ce_loss += ce_loss.item()
                    total_lwf_loss += lwf_loss.item()
                else:
                    outputs = self.model(batch_features)
                    total_batch_loss = criterion(outputs, batch_labels)
                    total_ce_loss += total_batch_loss.item()

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += total_batch_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            scheduler.step()
            avg_lwf_loss = total_lwf_loss / len(dataloader) if total_lwf_loss > 0 else 0
            epoch_lwf_losses.append(avg_lwf_loss)

            if (epoch + 1) % 5 == 0:
                accuracy = 100 * correct / total
                if self.learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
                    print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
                          f"CE={total_ce_loss/len(dataloader):.4f}, "
                          f"LwF={avg_lwf_loss:.4f}, Acc={accuracy:.2f}%")
                else:
                    print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Acc={accuracy:.2f}%")

        self.lwf_losses.extend(epoch_lwf_losses)
        training_time = time.time() - start_time
        self.trained_persons.extend(batch_persons)

        return training_time

    def evaluate_model(self):
        """Evaluate model performance"""
        if not self.model or not self.trained_persons:
            return {}

        start_time = time.time()

        # Prepare test data
        test_images = []
        true_labels = []

        for i, person in enumerate(self.trained_persons):
            if person in self.test_data:
                test_images.append(self.test_data[person]['test_image'])
                true_labels.append(i)

        if not test_images:
            return {}

        # Extract features and predict
        test_features = self.extract_features_batch(test_images, use_cache=True)
        valid_indices = [i for i, feat in enumerate(test_features) if np.any(feat)]
        
        if len(valid_indices) < len(test_features):
            test_features = test_features[valid_indices]
            true_labels = [true_labels[i] for i in valid_indices]

        if len(test_features) == 0:
            return {}

        test_features_tensor = torch.FloatTensor(test_features).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_features_tensor)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0].cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

        # Calculate FAR and FRR
        cm = confusion_matrix(true_labels, predicted_labels)
        n_classes = len(self.trained_persons)

        if n_classes > 1:
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            far = np.mean(fp / (fp + tn + 1e-8))
            frr = np.mean(fn / (fn + tp + 1e-8))
        else:
            far = 0.0
            frr = 1.0 - accuracy

        execution_time = time.time() - start_time
        model_size = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'execution_time': execution_time,
            'model_size': model_size,
            'num_persons': len(self.trained_persons),
            'avg_confidence': np.mean(max_probs),
            'cache_hits': len(self.feature_cache),
            'avg_lwf_loss': np.mean(self.lwf_losses) if self.lwf_losses else 0.0,
            'model_type': self.model_type.value,
            'learning_strategy': self.learning_strategy.value
        }

    def save_model(self, path):
        """Save the trained model"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'trained_persons': self.trained_persons,
                'num_classes': len(self.trained_persons),
                'model_type': self.model_type.value,
                'learning_strategy': self.learning_strategy.value,
                'lwf_weight': self.lwf_weight,
                'temperature': self.temperature,
                'lwf_losses': self.lwf_losses,
                'feature_cache_size': len(self.feature_cache)
            }, path)
            print(f"Model saved to {path}")

    def run_incremental_experiment(self):
        """Run the complete incremental learning experiment"""
        print(f"Starting Incremental Face Recognition Experiment")
        print(f"Model: {self.model_type.value}, Strategy: {self.learning_strategy.value}")
        print("=" * 80)

        # Organize dataset
        self.organize_dataset()

        if not self.person_folders:
            print("Error: No person folders found in dataset")
            return pd.DataFrame()

        # Create batches
        batches = [self.person_folders[i:i+self.batch_size]
                  for i in range(0, len(self.person_folders), self.batch_size)]

        print(f"Created {len(batches)} batches with batch size {self.batch_size}")
        if self.learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
            print(f"LwF Configuration: Alpha={self.lwf_weight}, Temperature={self.temperature}")

        results_df = pd.DataFrame()

        for batch_idx, batch_persons in enumerate(batches):
            print(f"\n{'='*35} BATCH {batch_idx + 1} {'='*35}")
            print(f"Processing persons: {batch_persons}")

            # Train on current batch
            is_first = (batch_idx == 0)
            if batch_idx >= 1 and self.learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
                print(f"LwF ACTIVE: Alpha={self.lwf_weight}, Temperature={self.temperature}")
            elif batch_idx >= 1 and self.learning_strategy == LearningStrategy.TRANSFER_JOINT:
                print(f"Joint Training ACTIVE: Using old + new data")
            elif batch_idx == 0:
                print(f"First batch - establishing baseline knowledge")

            training_time = self.train_batch(batch_persons, is_first_batch=is_first)

            if training_time == 0:
                print(f"Skipping batch {batch_idx + 1} due to training issues")
                continue

            # Evaluate model
            metrics = self.evaluate_model()
            if not metrics:
                print(f"Skipping batch {batch_idx + 1} due to evaluation issues")
                continue

            metrics['batch'] = batch_idx + 1
            metrics['training_time'] = training_time
            metrics['persons_in_batch'] = len(batch_persons)

            # Save model
            model_name = f"{self.model_type.value}_{self.learning_strategy.value}"
            model_path = f'{model_name}_model_batch_{batch_idx + 1}.pth'
            self.save_model(model_path)

            # Store results
            self.metrics_history.append(metrics)

            # Print current results
            print(f"\nBatch {batch_idx + 1} Results:")
            print(f"Persons trained: {metrics['num_persons']}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"FAR: {metrics['far']:.4f}")
            print(f"FRR: {metrics['frr']:.4f}")
            print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
            print(f"Training Time: {metrics['training_time']:.2f}s")
            print(f"Execution Time: {metrics['execution_time']:.4f}s")
            print(f"Model Size: {metrics['model_size']:.2f} MB")
            if self.learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
                print(f"Average LwF Loss: {metrics['avg_lwf_loss']:.4f}")
            print(f"Feature Cache Hits: {metrics['cache_hits']}")

        # Create results DataFrame
        if self.metrics_history:
            results_df = pd.DataFrame(self.metrics_history)

            # Save results
            results_filename = f'{model_name}_incremental_learning_results.csv'
            results_df.to_csv(results_filename, index=False)
            print(f"\nResults saved to {results_filename}")
        else:
            print("Warning: No successful batches processed")

        return results_df

    def plot_results(self, results_df):
        """Plot experiment results"""
        if results_df.empty:
            print("No results to plot")
            return

        model_name = f"{self.model_type.value}_{self.learning_strategy.value}"
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Main title
        fig.suptitle(f'{self.model_type.value.upper()} with {self.learning_strategy.value.upper()}', 
                     fontsize=16, fontweight='bold')

        # Accuracy over batches
        axes[0,0].plot(results_df['batch'], results_df['accuracy'], 'b-o', linewidth=2, markersize=8)
        axes[0,0].set_title('Accuracy vs Batch', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Batch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(0, 1)

        # F1-Score over batches
        axes[0,1].plot(results_df['batch'], results_df['f1_score'], 'g-o', linewidth=2, markersize=8)
        axes[0,1].set_title('F1-Score vs Batch', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Batch')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim(0, 1)

        # FAR and FRR
        axes[0,2].plot(results_df['batch'], results_df['far'], 'r-o', label='FAR', linewidth=2, markersize=8)
        axes[0,2].plot(results_df['batch'], results_df['frr'], 'orange', marker='s', label='FRR', linewidth=2, markersize=8)
        axes[0,2].set_title('FAR and FRR vs Batch', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Batch')
        axes[0,2].set_ylabel('Rate')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # LwF Loss (if applicable)
        if 'avg_lwf_loss' in results_df.columns and results_df['avg_lwf_loss'].sum() > 0:
            axes[1,0].plot(results_df['batch'], results_df['avg_lwf_loss'], 'purple', marker='d', linewidth=2, markersize=8)
            axes[1,0].set_title('LwF Loss vs Batch', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Batch')
            axes[1,0].set_ylabel('LwF Loss')
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No LwF Loss\n(Strategy does not use LwF)', 
                          ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('LwF Loss vs Batch', fontsize=14, fontweight='bold')

        # Training Time
        axes[1,1].plot(results_df['batch'], results_df['training_time'], 'brown', marker='^', linewidth=2, markersize=8)
        axes[1,1].set_title('Training Time vs Batch', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Batch')
        axes[1,1].set_ylabel('Time (seconds)')
        axes[1,1].grid(True, alpha=0.3)

        # Model Size
        axes[1,2].plot(results_df['batch'], results_df['model_size'], 'teal', marker='h', linewidth=2, markersize=8)
        axes[1,2].set_title('Model Size vs Batch', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Batch')
        axes[1,2].set_ylabel('Size (MB)')
        axes[1,2].grid(True, alpha=0.3)

        # Knowledge Retention Analysis
        if len(results_df) > 1:
            first_accuracy = results_df['accuracy'].iloc[0]
            accuracy_retention = results_df['accuracy'] / first_accuracy
            axes[2,0].plot(results_df['batch'], accuracy_retention, 'darkgreen', marker='o', linewidth=2, markersize=8)
            axes[2,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Retention')
            axes[2,0].set_title('Knowledge Retention', fontsize=14, fontweight='bold')
            axes[2,0].set_xlabel('Batch')
            axes[2,0].set_ylabel('Retention Ratio')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)

        # Confidence vs Accuracy
        scatter = axes[2,1].scatter(results_df['avg_confidence'], results_df['accuracy'],
                         c=results_df['batch'], cmap='viridis', s=100, alpha=0.7)
        axes[2,1].set_title('Confidence vs Accuracy', fontsize=14, fontweight='bold')
        axes[2,1].set_xlabel('Average Confidence')
        axes[2,1].set_ylabel('Accuracy')
        axes[2,1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2,1], label='Batch')

        # Scalability Analysis
        axes[2,2].plot(results_df['num_persons'], results_df['accuracy'], 'navy', marker='h', linewidth=2, markersize=8)
        axes[2,2].set_title('Scalability: Accuracy vs Persons', fontsize=14, fontweight='bold')
        axes[2,2].set_xlabel('Number of Persons')
        axes[2,2].set_ylabel('Accuracy')
        axes[2,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filename = f'{model_name}_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved to {plot_filename}")

def get_available_models():
    """Get list of available models"""
    available = []
    for model_type in ModelType:
        if MODEL_CONFIGS[model_type]['available']:
            available.append(model_type)
    return available

def print_framework_summary():
    """Print framework capabilities summary"""
    available_models = get_available_models()
    
    print(f"Unified Face Recognition Incremental Learning Framework")
    print(f"{'='*80}")
    print(f"Available Models: {len(available_models)}")
    
    for model in available_models:
        config = MODEL_CONFIGS[model]
        print(f"   • {model.value}: {config['description']}")
    
    if not VGGFACE2_AVAILABLE:
        print(f"   Warning: VGGFace2 unavailable (install: pip install facenet-pytorch)")
    
    print(f"\nLearning Strategies: {len(LearningStrategy)}")
    for strategy in LearningStrategy:
        print(f"   • {strategy.value}")
    
    total_configs = len(available_models) * len(LearningStrategy)
    print(f"\nTotal Configurations: {total_configs}")
    print(f"Advanced incremental learning with catastrophic forgetting mitigation")

def run_experiment(dataset_path, model_type, learning_strategy, batch_size=5, lwf_weight=0.5, temperature=3.0):
    """Run a single experiment configuration"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {model_type.value.upper()} + {learning_strategy.value.upper()}")
    print(f"{'='*80}")
    
    # Initialize researcher
    researcher = UnifiedFaceRecognitionResearcher(
        dataset_path=dataset_path,
        model_type=model_type,
        learning_strategy=learning_strategy,
        batch_size=batch_size,
        lwf_weight=lwf_weight,
        temperature=temperature
    )
    
    # Run experiment
    results = researcher.run_incremental_experiment()
    
    if not results.empty:
        # Plot results
        researcher.plot_results(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Configuration: {model_type.value} + {learning_strategy.value}")
        print(f"Total persons trained: {results['num_persons'].iloc[-1]}")
        print(f"Final accuracy: {results['accuracy'].iloc[-1]:.4f}")
        print(f"Average training time per batch: {results['training_time'].mean():.2f}s")
        print(f"Final model size: {results['model_size'].iloc[-1]:.2f} MB")
        print(f"Final FAR: {results['far'].iloc[-1]:.4f}")
        print(f"Final FRR: {results['frr'].iloc[-1]:.4f}")
        print(f"Final confidence: {results['avg_confidence'].iloc[-1]:.4f}")
        
        if learning_strategy in [LearningStrategy.TRANSFER_JOINT_LWF, LearningStrategy.TRANSFER_LWF_ONLY]:
            print(f"Average LwF loss: {results['avg_lwf_loss'].mean():.4f}")
            
            # Forgetting analysis
            if len(results) > 1:
                max_accuracy = results['accuracy'].max()
                final_accuracy = results['accuracy'].iloc[-1]
                forgetting_amount = max_accuracy - final_accuracy
                print(f"Forgetting amount: {forgetting_amount:.4f}")
                print(f"Forgetting percentage: {(forgetting_amount/max_accuracy)*100:.2f}%")
        
        print(f"Feature cache efficiency: {results['cache_hits'].iloc[-1]} cached features")
        
        return results, researcher
    else:
        print("ERROR: Experiment failed - check dataset and configuration")
        return None, None

def run_comparative_study(dataset_path, batch_size=5):
    """Run all available experiment configurations for comprehensive comparison"""
    
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE COMPARATIVE STUDY")
    print(f"{'='*100}")
    print(f"Dataset: {dataset_path}")
    print(f"Batch Size: {batch_size}")
    
    available_models = get_available_models()
    all_strategies = list(LearningStrategy)
    
    print(f"Available Models: {[m.value for m in available_models]}")
    print(f"Learning Strategies: {[s.value for s in all_strategies]}")
    
    all_results = {}
    
    # Run all combinations
    for model_type in available_models:
        for learning_strategy in all_strategies:
            try:
                print(f"\nStarting: {model_type.value} + {learning_strategy.value}")
                results, researcher = run_experiment(
                    dataset_path=dataset_path,
                    model_type=model_type,
                    learning_strategy=learning_strategy,
                    batch_size=batch_size
                )
                
                if results is not None:
                    experiment_name = f"{model_type.value}_{learning_strategy.value}"
                    all_results[experiment_name] = {
                        'results': results,
                        'researcher': researcher,
                        'model_type': model_type,
                        'learning_strategy': learning_strategy
                    }
                    print(f"Completed: {experiment_name}")
                else:
                    print(f"Failed: {model_type.value} + {learning_strategy.value}")
                    
            except Exception as e:
                print(f"Error in {model_type.value} + {learning_strategy.value}: {e}")
                continue
    
    # Generate comparative analysis
    if all_results:
        generate_comparative_analysis(all_results)
    
    return all_results

def generate_comparative_analysis(all_results):
    """Generate comprehensive comparative analysis"""
    
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    # Create comparison DataFrame
    comparison_data = []
    for exp_name, exp_data in all_results.items():
        results = exp_data['results']
        final_metrics = results.iloc[-1]
        
        comparison_data.append({
            'Experiment': exp_name,
            'Model': exp_data['model_type'].value,
            'Strategy': exp_data['learning_strategy'].value,
            'Final_Accuracy': final_metrics['accuracy'],
            'Final_F1': final_metrics['f1_score'],
            'Final_FAR': final_metrics['far'],
            'Final_FRR': final_metrics['frr'],
            'Avg_Training_Time': results['training_time'].mean(),
            'Final_Model_Size': final_metrics['model_size'],
            'Avg_LwF_Loss': final_metrics.get('avg_lwf_loss', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_df.to_csv('comprehensive_comparison.csv', index=False)
    print("\nDetailed comparison saved to 'comprehensive_comparison.csv'")
    
    # Print summary table
    print("\nPERFORMANCE SUMMARY:")
    print(comparison_df[['Experiment', 'Final_Accuracy', 'Final_F1', 'Final_Model_Size', 'Avg_Training_Time']].round(4))
    
    # Best performers
    print(f"\nBEST PERFORMERS:")
    print(f"Highest Accuracy: {comparison_df.loc[comparison_df['Final_Accuracy'].idxmax(), 'Experiment']} "
          f"({comparison_df['Final_Accuracy'].max():.4f})")
    print(f"Smallest Model: {comparison_df.loc[comparison_df['Final_Model_Size'].idxmin(), 'Experiment']} "
          f"({comparison_df['Final_Model_Size'].min():.2f} MB)")
    print(f"Fastest Training: {comparison_df.loc[comparison_df['Avg_Training_Time'].idxmin(), 'Experiment']} "
          f"({comparison_df['Avg_Training_Time'].min():.2f}s)")
    
    # Create comparative plots
    create_comparative_plots(comparison_df, all_results)

def create_comparative_plots(comparison_df, all_results):
    """Create comparative visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Experimental Comparison', fontsize=16, fontweight='bold')
    
    # Color mapping for models
    model_colors = {
        'mobilefacenet': 'blue',
        'edgeface': 'red',
        'vggface2': 'green'
    }
    
    # 1. Accuracy comparison
    colors = [model_colors.get(exp.split('_')[0], 'gray') for exp in comparison_df['Experiment']]
    axes[0,0].bar(range(len(comparison_df)), comparison_df['Final_Accuracy'], color=colors)
    axes[0,0].set_title('Final Accuracy Comparison', fontweight='bold')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_xticks(range(len(comparison_df)))
    axes[0,0].set_xticklabels(comparison_df['Experiment'], rotation=45, ha='right')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Model size comparison
    axes[0,1].bar(range(len(comparison_df)), comparison_df['Final_Model_Size'], color=colors)
    axes[0,1].set_title('Model Size Comparison', fontweight='bold')
    axes[0,1].set_ylabel('Size (MB)')
    axes[0,1].set_xticks(range(len(comparison_df)))
    axes[0,1].set_xticklabels(comparison_df['Experiment'], rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Training time comparison
    axes[0,2].bar(range(len(comparison_df)), comparison_df['Avg_Training_Time'], color=colors)
    axes[0,2].set_title('Average Training Time Comparison', fontweight='bold')
    axes[0,2].set_ylabel('Time (seconds)')
    axes[0,2].set_xticks(range(len(comparison_df)))
    axes[0,2].set_xticklabels(comparison_df['Experiment'], rotation=45, ha='right')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Accuracy vs Model Size scatter
    for i, row in comparison_df.iterrows():
        model = row['Experiment'].split('_')[0]
        color = model_colors.get(model, 'gray')
        marker = 'o' if 'lwf' in row['Experiment'].lower() else '^'
        axes[1,0].scatter(row['Final_Model_Size'], row['Final_Accuracy'], 
                         c=color, marker=marker, s=100, alpha=0.7, label=row['Experiment'])
    
    axes[1,0].set_title('Accuracy vs Model Size Trade-off', fontweight='bold')
    axes[1,0].set_xlabel('Model Size (MB)')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Learning curves comparison
    for exp_name, exp_data in all_results.items():
        results = exp_data['results']
        model = exp_data['model_type'].value
        color = model_colors.get(model, 'gray')
        linestyle = '-' if 'lwf' in exp_name else '--'
        axes[1,1].plot(results['batch'], results['accuracy'], 
                      color=color, linestyle=linestyle, marker='o', label=exp_name, alpha=0.7)
    
    axes[1,1].set_title('Learning Curves Comparison', fontweight='bold')
    axes[1,1].set_xlabel('Batch')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Strategy effectiveness by model
    strategy_performance = comparison_df.groupby(['Model', 'Strategy'])['Final_Accuracy'].mean().unstack()
    strategy_performance.plot(kind='bar', ax=axes[1,2], color=['green', 'blue', 'purple', 'orange'])
    axes[1,2].set_title('Performance by Model and Strategy', fontweight='bold')
    axes[1,2].set_ylabel('Average Accuracy')
    axes[1,2].set_xlabel('Model')
    axes[1,2].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparative plots saved to 'comprehensive_comparison.png'")

def main():
    """Main function with unified argument handling"""
    
    parser = argparse.ArgumentParser(description='Unified Face Recognition Incremental Learning Framework')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--model', type=str, choices=[m.value for m in ModelType], 
                        help='Model type to use')
    parser.add_argument('--strategy', type=str, 
                        choices=[s.value for s in LearningStrategy],
                        help='Learning strategy to use')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for incremental learning (default: 5)')
    parser.add_argument('--lwf_weight', type=float, default=0.5,
                        help='LwF alpha weight for knowledge distillation (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for knowledge distillation (default: 3.0)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run comprehensive study with all configurations')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison between key configurations')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist!")
        return
    
    # Check model availability
    available_models = get_available_models()
    if args.model:
        model_type = ModelType(args.model)
        if model_type not in available_models:
            print(f"Error: Model {args.model} not available!")
            print(f"Available models: {[m.value for m in available_models]}")
            if args.model == 'vggface2' and not VGGFACE2_AVAILABLE:
                print("Install facenet-pytorch: pip install facenet-pytorch")
            return
    
    # Print framework summary
    print_framework_summary()
    print(f"Dataset: {args.dataset_path}")
    
    if args.run_all:
        # Run comprehensive comparative study
        print("Running comprehensive comparative study...")
        all_results = run_comparative_study(args.dataset_path, args.batch_size)
        print(f"Comprehensive study completed with {len(all_results)} experiments")
        
    elif args.compare:
        # Run key comparisons
        print("Running key configuration comparison...")
        
        comparison_configs = []
        for model in available_models:
            comparison_configs.extend([
                (model, LearningStrategy.TRANSFER_ONLY),
                (model, LearningStrategy.TRANSFER_JOINT_LWF)
            ])
        
        results = {}
        for model_type, learning_strategy in comparison_configs:
            print(f"\nRunning {model_type.value} + {learning_strategy.value}")
            result, researcher = run_experiment(
                dataset_path=args.dataset_path,
                model_type=model_type,
                learning_strategy=learning_strategy,
                batch_size=args.batch_size,
                lwf_weight=args.lwf_weight,
                temperature=args.temperature
            )
            if result is not None:
                results[f"{model_type.value}_{learning_strategy.value}"] = {
                    'results': result, 'researcher': researcher,
                    'model_type': model_type, 'learning_strategy': learning_strategy
                }
        
        if results:
            generate_comparative_analysis(results)
    
    elif args.model and args.strategy:
        # Run single experiment
        model_type = ModelType(args.model)
        learning_strategy = LearningStrategy(args.strategy)
        
        print(f"Running single experiment: {model_type.value} + {learning_strategy.value}")
        result, researcher = run_experiment(
            dataset_path=args.dataset_path,
            model_type=model_type,
            learning_strategy=learning_strategy,
            batch_size=args.batch_size,
            lwf_weight=args.lwf_weight,
            temperature=args.temperature
        )
        
        if result is not None:
            print("Single experiment completed successfully!")
        else:
            print("Single experiment failed!")
    
    else:
        # Interactive mode
        run_interactive_mode(args.dataset_path, args.batch_size, args.lwf_weight, args.temperature)

def run_interactive_mode(dataset_path, batch_size, lwf_weight, temperature):
    """Run interactive mode for experiment selection"""
    available_models = get_available_models()
    
    print("\nInteractive Mode - Choose your experiment:")
    
    # Generate menu options
    options = []
    option_num = 1
    
    for model in available_models:
        for strategy in LearningStrategy:
            print(f"{option_num}. {model.value.title()} + {strategy.value.replace('_', ' ').title()}")
            options.append((model, strategy))
            option_num += 1
    
    print(f"{option_num}. Run All Experiments (Comprehensive Study)")
    print(f"{option_num + 1}. Quick Comparison (Key experiments)")
    
    max_choice = option_num + 1
    
    try:
        choice = int(input(f"\nEnter your choice (1-{max_choice}): "))
        
        if choice == option_num:
            # Run all experiments
            print("Running comprehensive comparative study...")
            all_results = run_comparative_study(dataset_path, batch_size)
            print(f"Comprehensive study completed with {len(all_results)} experiments")
        
        elif choice == option_num + 1:
            # Quick comparison
            print("Running quick comparison...")
            quick_configs = []
            for model in available_models:
                quick_configs.extend([
                    (model, LearningStrategy.TRANSFER_ONLY),
                    (model, LearningStrategy.TRANSFER_JOINT_LWF)
                ])
            
            results = {}
            for model_type, learning_strategy in quick_configs:
                print(f"\nRunning {model_type.value} + {learning_strategy.value}")
                result, researcher = run_experiment(
                    dataset_path=dataset_path,
                    model_type=model_type,
                    learning_strategy=learning_strategy,
                    batch_size=batch_size,
                    lwf_weight=lwf_weight,
                    temperature=temperature
                )
                if result is not None:
                    results[f"{model_type.value}_{learning_strategy.value}"] = {
                        'results': result, 'researcher': researcher,
                        'model_type': model_type, 'learning_strategy': learning_strategy
                    }
            
            if results:
                generate_comparative_analysis(results)
        
        elif 1 <= choice <= len(options):
            # Single experiment
            model_type, learning_strategy = options[choice - 1]
            print(f"Running: {model_type.value} + {learning_strategy.value}")
            result, researcher = run_experiment(
                dataset_path=dataset_path,
                model_type=model_type,
                learning_strategy=learning_strategy,
                batch_size=batch_size,
                lwf_weight=lwf_weight,
                temperature=temperature
            )
            
            if result is not None:
                print("Experiment completed successfully!")
            else:
                print("Experiment failed!")
        
        else:
            print(f"Invalid choice! Please select 1-{max_choice}.")
            
    except ValueError:
        print(f"Invalid input! Please enter a number between 1-{max_choice}.")
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")

def print_usage_examples():
    """Print usage examples for the framework"""
    available_models = get_available_models()
    
    print(f"\n{'='*80}")
    print(f"USAGE EXAMPLES")
    print(f"{'='*80}")
    
    examples = [
        {
            'title': '1. Run single experiment',
            'command': f'python unified_face_recognition.py --dataset_path /path/to/dataset --model {available_models[0].value} --strategy transfer_joint_lwf',
            'description': f'Run {available_models[0].value.title()} with Transfer Learning + Joint Training + LwF'
        },
        {
            'title': '2. Run comprehensive study',
            'command': 'python unified_face_recognition.py --dataset_path /path/to/dataset --run_all',
            'description': f'Run all {len(available_models) * len(LearningStrategy)} experiment configurations for complete comparison'
        },
        {
            'title': '3. Run quick comparison',
            'command': 'python unified_face_recognition.py --dataset_path /path/to/dataset --compare',
            'description': 'Run key experiments for quick insight'
        },
        {
            'title': '4. Custom LwF parameters',
            'command': f'python unified_face_recognition.py --dataset_path /path/to/dataset --model {available_models[-1].value} --strategy transfer_lwf_only --lwf_weight 0.8 --temperature 4.0',
            'description': f'Run {available_models[-1].value.title()} with LwF only using custom parameters'
        },
        {
            'title': '5. Interactive mode',
            'command': 'python unified_face_recognition.py --dataset_path /path/to/dataset',
            'description': 'Run in interactive mode to choose experiment from menu'
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print(f"Command: {example['command']}")
        print(f"Description: {example['description']}")

def print_experiment_guide():
    """Print detailed guide about different experiments"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT STRATEGY GUIDE")
    print(f"{'='*80}")
    
    strategies = [
        {
            'name': 'Transfer Learning Only',
            'code': 'transfer_only',
            'description': 'Baseline approach - trains only on new classes',
            'pros': ['Fastest training', 'Minimal memory usage', 'Simple implementation'],
            'cons': ['High catastrophic forgetting', 'Poor retention of old classes'],
            'use_case': 'Baseline comparison, applications where forgetting is acceptable'
        },
        {
            'name': 'Transfer Learning + Joint Training',
            'code': 'transfer_joint',
            'description': 'Mixes old and new class data during training',
            'pros': ['Reduced forgetting', 'Good overall performance', 'Proven approach'],
            'cons': ['Requires storing old data', 'Increased training time'],
            'use_case': 'Standard incremental learning when storage is available'
        },
        {
            'name': 'Transfer Learning + Joint Training + LwF',
            'code': 'transfer_joint_lwf',
            'description': 'Combines data rehearsal with knowledge distillation',
            'pros': ['Best overall performance', 'Minimal forgetting', 'Stable learning'],
            'cons': ['Most complex', 'Highest computational cost'],
            'use_case': 'Production systems requiring highest accuracy'
        },
        {
            'name': 'Transfer Learning + LwF Only',
            'code': 'transfer_lwf_only',
            'description': 'Pure knowledge distillation without storing old data',
            'pros': ['Memory efficient', 'No old data storage', 'Good forgetting mitigation'],
            'cons': ['Lower performance than joint training', 'Requires careful tuning'],
            'use_case': 'Memory-constrained environments, privacy-sensitive applications'
        }
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']} ({strategy['code']})")
        print(f"   {strategy['description']}")
        print(f"   Pros: {', '.join(strategy['pros'])}")
        print(f"   Cons: {', '.join(strategy['cons'])}")
        print(f"   Use Case: {strategy['use_case']}")
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON GUIDE")
    print(f"{'='*80}")
    
    available_models = get_available_models()
    
    for model in available_models:
        config = MODEL_CONFIGS[model]
        print(f"\n{model.value.title()}")
        print(f"   {config['description']}")
        print(f"   Feature Dimension: {config['feature_dim']}")
        print(f"   Input Size: {config['input_size']}")
        if config['classifier_hidden']:
            print(f"   Classifier: {' → '.join(map(str, config['classifier_hidden']))} → classes")

def print_installation_guide():
    """Print installation requirements"""
    print(f"\n{'='*80}")
    print(f"INSTALLATION REQUIREMENTS")
    print(f"{'='*80}")
    print(f"Required packages:")
    print(f"pip install torch torchvision")
    print(f"pip install insightface onnxruntime")
    print(f"pip install opencv-python scikit-learn matplotlib pandas")
    print(f"pip install Pillow numpy")
    
    if not VGGFACE2_AVAILABLE:
        print(f"\nOptional (for VGGFace2 support):")
        print(f"pip install facenet-pytorch")
    else:
        print(f"\nAll dependencies available")

def print_dataset_requirements():
    """Print dataset structure requirements"""
    print(f"\n{'='*80}")
    print(f"DATASET REQUIREMENTS")
    print(f"{'='*80}")
    print(f"Dataset should be organized as:")
    print(f"dataset/")
    print(f"├── person1/")
    print(f"│   ├── image1.jpg")
    print(f"│   ├── image2.jpg")
    print(f"│   └── ...")
    print(f"├── person2/")
    print(f"│   └── ...")
    print(f"└── personN/")
    print(f"\nRequirements:")
    print(f"• Minimum: 2 images per person (1 for training, 1 for testing)")
    print(f"• Recommended: 3+ images per person for better results")
    print(f"• Supported formats: .jpg, .jpeg, .png, .bmp")
    print(f"• Images should contain clear faces")

# Utility functions for advanced users

def create_custom_experiment(dataset_path, custom_config):
    """Create and run custom experiment configuration"""
    print(f"Running custom experiment configuration...")
    
    # Validate model availability
    model_name = custom_config.get('model', 'mobilefacenet')
    try:
        model_type = ModelType(model_name)
        if not MODEL_CONFIGS[model_type]['available']:
            available_models = get_available_models()
            print(f"Model {model_name} not available. Using {available_models[0].value}")
            model_type = available_models[0]
    except ValueError:
        available_models = get_available_models()
        print(f"Invalid model {model_name}. Using {available_models[0].value}")
        model_type = available_models[0]
    
    # Validate strategy
    strategy_name = custom_config.get('strategy', 'transfer_joint_lwf')
    try:
        learning_strategy = LearningStrategy(strategy_name)
    except ValueError:
        print(f"Invalid strategy {strategy_name}. Using transfer_joint_lwf")
        learning_strategy = LearningStrategy.TRANSFER_JOINT_LWF
    
    return run_experiment(
        dataset_path=dataset_path,
        model_type=model_type,
        learning_strategy=learning_strategy,
        batch_size=custom_config.get('batch_size', 5),
        lwf_weight=custom_config.get('lwf_weight', 0.5),
        temperature=custom_config.get('temperature', 3.0)
    )

def run_parameter_sweep(dataset_path, param_ranges):
    """Run parameter sweep for hyperparameter optimization"""
    print(f"Running parameter sweep...")
    
    # Validate parameter ranges
    available_models = [m.value for m in get_available_models()]
    if 'model' in param_ranges:
        param_ranges['model'] = [m for m in param_ranges['model'] if m in available_models]
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_ranges.values()))
    param_names = list(param_ranges.keys())
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nParameter set {i+1}/{len(param_combinations)}: {dict(zip(param_names, params))}")
        
        config = dict(zip(param_names, params))
        result, researcher = create_custom_experiment(dataset_path, config)
        
        if result is not None:
            # Store parameter combination with results
            final_metrics = result.iloc[-1].copy()
            for param_name, param_value in config.items():
                final_metrics[f'param_{param_name}'] = param_value
            results.append(final_metrics)
    
    # Analyze parameter sweep results
    if results:
        sweep_df = pd.DataFrame(results)
        sweep_df.to_csv('parameter_sweep_results.csv', index=False)
        print(f"Parameter sweep results saved to 'parameter_sweep_results.csv'")
        
        # Find best parameters
        best_idx = sweep_df['accuracy'].idxmax()
        best_params = {col.replace('param_', ''): sweep_df.loc[best_idx, col] 
                      for col in sweep_df.columns if col.startswith('param_')}
        
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best accuracy: {sweep_df.loc[best_idx, 'accuracy']:.4f}")
    
    return results

def example_parameter_sweep(dataset_path):
    """Example of how to run parameter sweep"""
    available_models = [m.value for m in get_available_models()]
    
    param_ranges = {
        'model': available_models,
        'strategy': ['transfer_joint_lwf', 'transfer_lwf_only'],
        'lwf_weight': [0.3, 0.5, 0.8],
        'temperature': [2.0, 3.0, 4.0]
    }
    
    return run_parameter_sweep(dataset_path, param_ranges)

if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) == 1:
        # No arguments provided - show help
        print_framework_summary()
        
        print(f"\n{'='*80}")
        print(f"GETTING STARTED")
        print(f"{'='*80}")
        print(f"This framework requires a dataset path. Here are your options:")
        print(f"\n1. Command Line Usage:")
        print(f"   python unified_face_recognition.py --dataset_path /path/to/dataset [options]")
        print(f"\n2. Interactive Mode:")
        print(f"   python unified_face_recognition.py --dataset_path /path/to/dataset")
        
        print_usage_examples()
        print_experiment_guide()
        print_dataset_requirements()
        print_installation_guide()
        
        # Ask if user wants to continue interactively
        try:
            dataset_path = input(f"\nEnter dataset path to continue interactively (or Ctrl+C to exit): ").strip()
            if dataset_path and os.path.exists(dataset_path):
                # Run in interactive mode
                sys.argv.extend(['--dataset_path', dataset_path])
                main()
            else:
                print(f"Invalid dataset path: {dataset_path}")
        except KeyboardInterrupt:
            print(f"\nGoodbye!")
    else:
        # Arguments provided - run normally
        main()