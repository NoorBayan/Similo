import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def print_comparative_table(all_results):
    """طباعة جدول المقارنة الشامل جاهز للورقة البحثية"""
    # تصفية البيانات الخام من القاموس للعرض
    display_dict = {}
    for model, metrics in all_results.items():
        if model != 'TF-IDF (Baseline)': # يمكن استبعاد البيس لاين أو إبقائه حسب الرغبة
            display_dict[model] = {
                'Accuracy': metrics['Accuracy'],
                'Macro-F1': metrics['Macro_F1'],
                'Train Time (s)': metrics['Train_Time (s)'],
                'Inference Time (s)': metrics['Inference_Time (s)']
            }
    
    df = pd.DataFrame(display_dict).T
    print("\n" + "="*80)
    print("🏆 Overall Performance Comparison (Mean ± Std over 3 seeds)")
    print("="*80)
    print(df.to_markdown())
    print("="*80 + "\n")

def generate_reports(predictions_output, test_df, label_encoder, model_name):
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    target_names = label_encoder.classes_

    print(f"\n{'='*50}\n📊 Best Seed Classification Report: {model_name}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix ({model_name})", pad=15)
    plt.ylabel('Actual Pragmatic Function')
    plt.xlabel('Predicted Pragmatic Function')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
