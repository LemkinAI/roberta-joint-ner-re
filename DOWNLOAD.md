# 📥 Download RoBERTa Joint NER+RE Model

The large model files are hosted on HuggingFace Hub for better performance and reliability.

## 🤗 Direct Download from HuggingFace Hub

### Option 1: Using Transformers Library (Recommended)
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Automatically downloads and caches the model
tokenizer = AutoTokenizer.from_pretrained("lemkin-ai/roberta-joint-ner-re")
model = AutoModelForTokenClassification.from_pretrained("lemkin-ai/roberta-joint-ner-re")
```

### Option 2: Manual Download via CLI
```bash
# Install HuggingFace Hub CLI
pip install huggingface_hub

# Download all model files
huggingface-cli download lemkin-ai/roberta-joint-ner-re --local-dir ./models/roberta-joint-ner-re/
```

### Option 3: Git Clone from HuggingFace
```bash
# Clone the model repository
git clone https://huggingface.co/lemkin-ai/roberta-joint-ner-re
```

## 📊 Model Files Available on HuggingFace Hub

| File | Size | Description |
|------|------|-------------|
| `pytorch_model.bin` | 2.1GB | PyTorch model weights |
| `config.json` | 2KB | Model configuration |
| `tokenizer_config.json` | 1KB | Tokenizer configuration |
| `vocab.json` | 779KB | Vocabulary file |
| `merges.txt` | 446KB | BPE merges |
| `tokenizer.json` | 2.0MB | Fast tokenizer |
| `special_tokens_map.json` | 1KB | Special tokens mapping |

## 🌐 Model Hub URL
**https://huggingface.co/lemkin-ai/roberta-joint-ner-re**

## ⚡ Quick Start
```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model
tokenizer = AutoTokenizer.from_pretrained("lemkin-ai/roberta-joint-ner-re")
model = AutoModelForTokenClassification.from_pretrained("lemkin-ai/roberta-joint-ner-re")

# Example usage
text = "The International Criminal Court issued a warrant for the general's arrest."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Process predictions for NER + RE tasks
print("Entity predictions:", predictions)
```

---
*For complete documentation, see the main repository README and model card on HuggingFace Hub.* 