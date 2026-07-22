import numpy as np
import time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers import set_seed
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

def run_comparative_study(model_name, train_df, test_df, num_labels, seeds=[42, 123, 2024]):
    """
    تقوم هذه الدالة بتشغيل النموذج عدة مرات ببذور عشوائية مختلفة لضمان موثوقية النتائج.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']].reset_index(drop=True)).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df[['clean_text', 'label']].reset_index(drop=True)).map(tokenize_function, batched=True)

    classes = np.unique(train_df['label'])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])

    results_list = []
    
    for seed in seeds:
        print(f"   -> تشغيل البذرة العشوائية (Seed): {seed} ...")
        set_seed(seed) # توحيد العشوائية
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="single_label_classification"
        )

        training_args = TrainingArguments(
            output_dir=f"./temp_{seed}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=10,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            save_total_limit=1,
            seed=seed,
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

        # 1. قياس وقت التدريب
        start_train = time.time()
        trainer.train()
        train_time = time.time() - start_train

        # 2. قياس وقت الاستدلال (Inference)
        start_inf = time.time()
        eval_results = trainer.evaluate()
        inf_time = time.time() - start_inf
        
        # تخزين التنبؤات لأجل مصفوفة الارتباك (نحتفظ بآخر تنبؤ فقط أو الأفضل)
        predictions = trainer.predict(test_ds)

        results_list.append({
            'accuracy': eval_results['eval_accuracy'],
            'macro_f1': eval_results['eval_macro_f1'],
            'train_time': train_time,
            'inference_time': inf_time,
            'preds': predictions # للاستخدام في التحليل
        })

    # حساب المتوسط والانحراف المعياري
    acc_mean = np.mean([r['accuracy'] for r in results_list]) * 100
    acc_std = np.std([r['accuracy'] for r in results_list]) * 100
    f1_mean = np.mean([r['macro_f1'] for r in results_list]) * 100
    f1_std = np.std([r['macro_f1'] for r in results_list]) * 100
    t_train_mean = np.mean([r['train_time'] for r in results_list])
    t_inf_mean = np.mean([r['inference_time'] for r in results_list])

    final_metrics = {
        'Accuracy': f"{acc_mean:.2f} ±{acc_std:.2f}",
        'Macro_F1': f"{f1_mean:.2f} ±{f1_std:.2f}",
        'Train_Time (s)': f"{t_train_mean:.1f}",
        'Inference_Time (s)': f"{t_inf_mean:.2f}",
        'Raw_Data': results_list # نحتفظ بالبيانات الخام إذا احتجناها
    }

    # نرجع أفضل تنبؤ لرسم مصفوفة الارتباك وتحليل الأخطاء لاحقاً
    best_preds = max(results_list, key=lambda x: x['macro_f1'])['preds']

    return final_metrics, best_preds
