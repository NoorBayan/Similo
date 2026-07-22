from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def run_baseline(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
    clf.fit(X_train, train_df['label'])
    preds = clf.predict(X_test)

    acc = accuracy_score(test_df['label'], preds)
    f1 = f1_score(test_df['label'], preds, average='macro')
    
    return {'Accuracy': acc, 'Macro_F1': f1}
