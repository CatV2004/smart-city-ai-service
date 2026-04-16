import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.model.text_classifier import UrbanTextClassifier
from app.config import settings

# Khởi tạo classifier với đường dẫn từ settings
classifier = UrbanTextClassifier(
    model_path=settings.TEXT_MODEL_PATH,
    tokenizer_path=settings.TEXT_TOKENIZER_PATH
)

# Test
test_texts = [
    "ổ gà to trước cổng trường",
    "đường nứt toác dài 3 mét",
    "rác thải bốc mùi hôi thối",
    "cây đổ chắn ngang đường",
    "cột điện nghiêng sắp đổ",
    "biển báo giao thông bị gãy",
    "tường bị vẽ bậy lem nhem"
]

print("=" * 50)
print("TEST URBAN TEXT CLASSIFIER")
print("=" * 50)

for text in test_texts:
    result = classifier.predict(text, return_all_probs=True)
    print(f"\nText: {text}")
    print(f"  → {result['label_vi']} (conf: {result['confidence']:.4f})")
    print(f"  Top 3: {[p['label_vi'] for p in result['top_predictions']]}")

print("\n" + "=" * 50)
print("✅ Test completed!")