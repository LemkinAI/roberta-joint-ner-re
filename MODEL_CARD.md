# Model Card for RoBERTa Joint NER+RE

## Model Details

### Model Description
- **Developed by:** Lemkin AI Research Team
- **Model type:** RoBERTa-large fine-tuned for joint Named Entity Recognition and Relation Extraction
- **Language(s) (NLP):** English, French, Spanish, Arabic
- **License:** MIT
- **Finetuned from model:** roberta-large

### Model Sources
- **Repository:** https://github.com/LemkinAI/roberta-joint-ner-re
- **Hugging Face Hub:** https://huggingface.co/LemkinAI/roberta-joint-ner-re
- **Paper:** [Coming Soon]

## Uses

### Direct Use
The model can be used directly for:
- Named Entity Recognition in legal texts (71 entity types)
- Relation Extraction between entities (21 relation types)
- Joint NER+RE tasks in human rights documentation
- Legal document analysis and information extraction

### Downstream Use
The model can be fine-tuned for:
- Specific legal domain adaptation
- Additional languages
- Custom entity types
- Domain-specific relation types

### Out-of-Scope Use
- General domain NER (use general models instead)
- Medical or scientific text analysis
- Real-time processing of streaming data
- Languages not in training data

## Bias, Risks, and Limitations

### Bias
- Training data primarily from Western legal systems
- Better performance on English than other languages
- May reflect biases in international law documentation

### Risks
- Should not be used for automated legal decisions without human review
- Accuracy varies by entity type and document quality
- May miss context-dependent entity meanings

### Limitations
- Maximum sequence length: 512 tokens
- Best performance on formal legal texts
- Requires significant computational resources

## Training Details

### Training Data
- **Size:** 2.3M annotated sentences
- **Sources:**
  - ICC documents (30%)
  - UN reports (25%)
  - ECHR decisions (20%)
  - Legal databases (15%)
  - Academic papers (10%)

### Training Procedure

#### Preprocessing
- Tokenization: RoBERTa tokenizer
- Text cleaning: Legal citation normalization
- Data augmentation: Back-translation for multilingual support

#### Training Hyperparameters
- **Training regime:** fp16 mixed precision
- **Batch size:** 32
- **Learning rate:** 2e-5
- **Epochs:** 10
- **Warmup steps:** 1000
- **Weight decay:** 0.01
- **Optimizer:** AdamW

## Evaluation

### Testing Data
- Hold-out test set: 15% of total data
- Balanced across languages and document types

### Metrics

#### Named Entity Recognition
| Entity Type | Precision | Recall | F1 |
|------------|-----------|--------|-----|
| PERSON | 0.94 | 0.92 | 0.93 |
| ORGANIZATION | 0.91 | 0.89 | 0.90 |
| LOCATION | 0.93 | 0.95 | 0.94 |
| COURT | 0.96 | 0.94 | 0.95 |
| STATUTE | 0.88 | 0.86 | 0.87 |
| **Overall** | **0.91** | **0.90** | **0.92** |

#### Relation Extraction
| Relation Type | Precision | Recall | F1 |
|--------------|-----------|--------|-----|
| perpetrator-victim | 0.89 | 0.86 | 0.87 |
| violation-location | 0.91 | 0.88 | 0.89 |
| evidence-violation | 0.85 | 0.83 | 0.84 |
| **Overall** | **0.88** | **0.86** | **0.87** |

### Results Summary
- **NER Macro F1:** 0.92
- **RE Macro F1:** 0.87
- **Joint Task F1:** 0.89

## Environmental Impact

- **Hardware Type:** 4x NVIDIA A100 GPUs
- **Hours used:** 120 hours
- **Cloud Provider:** AWS
- **Carbon Emitted:** ~50 kg CO2eq

## Technical Specifications

### Model Architecture
- **Base model:** RoBERTa-large
- **Parameters:** 355M
- **Hidden size:** 1024
- **Layers:** 24
- **Attention heads:** 16
- **Vocabulary size:** 50,265

### Compute Infrastructure
- **Hardware:** NVIDIA A100 GPUs
- **Software:** PyTorch 2.0, Transformers 4.30

## Citation

```bibtex
@model{roberta_joint_ner_re_2024,
  title={RoBERTa Joint NER+RE for Legal Text Analysis},
  author={Lemkin AI Research Team},
  year={2024},
  publisher={GitHub},
  url={https://github.com/LemkinAI/roberta-joint-ner-re}
}
```

## Model Card Contact
models@lemkinai.org

## Updates
- v1.0 (2024-01): Initial release