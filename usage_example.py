#!/usr/bin/env python3
"""
Example usage script for RoBERTa Joint NER+RE Model
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

def load_model():
    """Load the RoBERTa joint NER+RE model"""
    model_name = "LemkinAI/roberta-joint-ner-re"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return tokenizer, model

def extract_entities_pipeline(text):
    """Extract entities using the pipeline approach"""
    # Create NER pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model="LemkinAI/roberta-joint-ner-re",
        tokenizer="LemkinAI/roberta-joint-ner-re",
        aggregation_strategy="simple"
    )

    entities = ner_pipeline(text)
    return entities

def extract_entities_manual(text, tokenizer, model):
    """Extract entities using manual processing"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_labels = torch.argmax(predictions, dim=-1)

    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[label_id.item()] for label_id in predicted_labels[0]]

    # Group entities
    entities = []
    current_entity = None

    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text': token.replace('Ġ', ' '),
                'label': label[2:],
                'start': len(entities)
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text'] += token.replace('Ġ', ' ')
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

def main():
    """Main example function"""
    # Example legal texts
    examples = [
        "The International Criminal Court issued a warrant for war crimes in Syria.",
        "Civilians were targeted during the siege of Aleppo in 2016.",
        "Military forces conducted arbitrary arrests of civilians in Darfur.",
        "The Court found evidence of torture at Abu Ghraib prison.",
        "Medical facilities were targeted despite protected status under Geneva Conventions."
    ]

    print("RoBERTa Joint NER+RE Model Example")
    print("=" * 50)

    # Method 1: Using pipeline (recommended)
    print("\nMethod 1: Using Pipeline")
    print("-" * 30)

    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}: {text}")
        entities = extract_entities_pipeline(text)

        if entities:
            for entity in entities:
                print(f"  • {entity['word']} ({entity['entity_group']}) - Score: {entity['score']:.3f}")
        else:
            print("  No entities found")

    # Method 2: Manual processing
    print("\n\nMethod 2: Manual Processing")
    print("-" * 30)

    tokenizer, model = load_model()

    for i, text in enumerate(examples[:2], 1):  # Only first 2 examples for brevity
        print(f"\nExample {i}: {text}")
        entities = extract_entities_manual(text, tokenizer, model)

        if entities:
            for entity in entities:
                print(f"  • {entity['text']} ({entity['label']})")
        else:
            print("  No entities found")

if __name__ == "__main__":
    main()