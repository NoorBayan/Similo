import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_model_comparison(results_dict):
    df_results = pd.DataFrame(results_dict).T.reset_index()
    df_results.rename(columns={'index': 'Model'}, inplace=True)
    
    df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette=["#4C72B0", "#DD8452"])
    plt.ylim(0, 1.0)
    plt.title("Performance Comparison Models", pad=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
    plt.show()

def generate_reports(predictions_output, test_df, label_encoder, model_name):
    y_pred = np.argmax(predictions_output.predictions, axis=1)
    y_true = predictions_output.label_ids
    target_names = label_encoder.classes_

    print(f"\n{'='*50}\n📊 Classification Report: {model_name}\n{'='*50}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix ({model_name})", pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Error Analysis
    errors_df = pd.DataFrame({
        'Verse Text': test_df['text'].values,
        'Actual Intent': [target_names[i] for i in y_true],
        'Predicted Intent': [target_names[i] for i in y_pred]
    })
    misclassified = errors_df[errors_df['Actual Intent'] != errors_df['Predicted Intent']]
    
    print(f"\n{'='*50}\n🔍 Sample Errors ({model_name})\n{'='*50}")
    print(misclassified.sample(n=min(10, len(misclassified)), random_state=42).to_markdown(index=False))
