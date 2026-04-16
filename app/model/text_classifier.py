import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional

from app.model.labels import CLASS_NAMES, LABEL_VI


class PhoBERTClassifier(nn.Module):
    """PhoBERT Classifier - Model Architecture"""
    
    def __init__(
        self, 
        model_name: str = "vinai/phobert-base",
        num_classes: int = 7,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.hidden_size = self.config.hidden_size
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.dropout1,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            self.dropout2,
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.dropout3,
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        mean_pooled = torch.mean(last_hidden, dim=1)
        max_pooled = torch.max(last_hidden, dim=1)[0]
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        
        logits = self.classifier(pooled)
        return logits


class UrbanTextClassifier:
    """Text Classifier cho bài toán phản ánh đô thị"""
    
    def __init__(
        self, 
        model_path: str = "models/text/best_model.pt",
        tokenizer_path: str = "models/text/tokenizer",
        device: Optional[str] = None,
        max_length: int = 128
    ):
        # Xác định device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.max_length = max_length
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        elif 'id_to_label' in checkpoint:
            id_to_label = {int(k): v for k, v in checkpoint['id_to_label'].items()}
            self.class_names = [id_to_label[i] for i in range(len(id_to_label))]
        else:
            self.class_names = CLASS_NAMES
        
        self.num_classes = len(self.class_names)
        self.id_to_label = {i: label for i, label in enumerate(self.class_names)}
        self.label_vi = LABEL_VI
        
        # Lấy config
        self.config = checkpoint.get('config', {})
        model_name = self.config.get('model_name', 'vinai/phobert-base')
        dropout_rate = self.config.get('dropout_rate', 0.3)
        
        # Khởi tạo model
        self.model = PhoBERTClassifier(
            model_name=model_name,
            num_classes=self.num_classes,
            dropout_rate=dropout_rate
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"✅ UrbanTextClassifier loaded on {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def predict(self, text: str, return_all_probs: bool = False) -> Dict:
        """Dự đoán label cho một text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probs = F.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)
        
        pred_label = self.id_to_label[pred_idx.item()]
        
        result = {
            'text': text,
            'label': pred_label,
            'label_vi': self.label_vi.get(pred_label, pred_label),
            'confidence': confidence.item()
        }
        
        if return_all_probs:
            result['probabilities'] = {
                self.id_to_label[i]: probs[0, i].item()
                for i in range(self.num_classes)
            }
            
            top_k = min(3, self.num_classes)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            result['top_predictions'] = [
                {
                    'label': self.id_to_label[top_indices[0, i].item()],
                    'label_vi': self.label_vi.get(self.id_to_label[top_indices[0, i].item()], ''),
                    'confidence': top_probs[0, i].item()
                }
                for i in range(top_k)
            ]
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Dự đoán cho batch text"""
        return [self.predict(text) for text in texts]