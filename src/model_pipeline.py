import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from .custom_trainer import WeightedTrainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }

def train_and_evaluate_model(model_name, train_df, test_df, num_labels, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']].reset_index(drop=True)).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df[['clean_text', 'label']].reset_index(drop=True)).map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="single_label_classification"
    )

    # حساب أوزان الفئات للتعامل مع عدم التوازن
    classes = np.unique(train_df['label'])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none"
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    predictions = trainer.predict(test_ds)
    
    return results, predictions
