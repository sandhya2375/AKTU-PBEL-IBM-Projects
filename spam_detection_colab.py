# ============================================
# PROJECT 2: Text Analytics for Spam Detection using Naive Bayes
# IBM SkillsBuild - Google Colab Ready
# ============================================

# Step 1: Libraries install aur import karo
!pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import re

print("✅ Sab libraries import ho gaye!")

# ============================================
# Step 2: Sample Spam/Ham Dataset create karo
# ============================================
# Ye realistic messages hain - SMS spam dataset ke jaisa

data = {
    'message': [
        # Spam messages
        'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!',
        'U have a secret admirer who loves u. Cost £1.50 tone 1380. Sorry I will pay!',
        'Hey.. how are you? I am using same tutoring service since 6 months.',
        'FreeMsg: Hi resp. U will b contacted soon as we have a job with££ that matches your CV',
        'FREE ENTRY in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121',
        'Congratulations ur awarded a £500 Tango voucher/call 08714990521 claim code KL341. Valid 12 hours only.',
        'Urgent! You have won a 1 week FREE membership in our £100,000 prize Jackpot!',
        'SIR PLEASE VISIT OUR WEBSITE FOR LOAN OFFER',
        'Claim your Prize!! Your details could be selected to win a brand new BMW, call now!',
        'You are a winner U have been specially selected to receive £1000 cash or a 4* holiday',
        
        # Ham messages
        'Hi Babe, how r u? I was thinking about u. Hope u are having a wonderful day!',
        'I am out of office will return on 15th of April. Will get back to you then.',
        'Did u manage to do the homework?',
        'Hey! Ready to go out tonight?',
        'Thank you for your business. We appreciate your order!',
        'Hey, can we catch up tomorrow?',
        'Just checking how you are doing these days!',
        'Meeting rescheduled to 3 PM tomorrow. Please confirm your attendance.',
        'Happy Birthday! Hope you have an amazing day!',
        'Let me know when you are free for a coffee.',
        'Thanks for the update. I will review it.',
        'Can you send me the report by end of day?',
        'Great! See you tomorrow at 5 PM.',
        'Do you want to grab lunch?',
        'I miss you. When are we meeting next?',
        'Your password has been reset. Ignore if you did not request this.',
        'Project deadline is extended to next week.',
        'Sorry, running late. Will be there in 10 mins.',
        'The meeting went well. Thanks for organizing!',
        'Can you help me with this code?',
        'PRIZE MONEY WAITING FOR YOU!!!!!!',
        'CLICK HERE TO GET FREE iPHONE NOW!!!',
        'Congratulations you won a Samsung Galaxy S21!',
        'LOTTERY WINNER ANNOUNCEMENT!!!',
        'GET RICH QUICK!!! EARN $5000 IN 2 WEEKS!!!',
        'Your account has been compromised. URGENT ACTION REQUIRED!',
        'SPECIAL OFFER: 50% OFF EVERYTHING TODAY ONLY!',
        'Hi there! Just wanted to check in with you.',
        'Thanks for meeting with me today.',
        'Looking forward to hearing from you!',
        'The project is on track. Good work team!',
        'Let me know if you need any help.',
        'See attached document for more details.',
        'Schedule changed. New time is 2 PM.',
        'Great job on the presentation!',
        'Can we discuss this over a call?',
        'I will get back to you soon.',
    ],
    'label': [
        # Spam (1)
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        # Ham (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # More spam
        1, 1, 1, 1, 1, 1, 1,
        # More ham
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
}

df = pd.DataFrame(data)

print("✅ Dataset create ho gaya!")
print("\nDataset shape:", df.shape)
print("\nFirst 5 messages:")
print(df.head())
print("\nSpam vs Ham distribution:")
print(df['label'].value_counts())
print("\n0 = HAM (legitimate), 1 = SPAM")

# ============================================
# Step 3: TEXT PREPROCESSING
# ============================================
print("\n" + "="*50)
print("TEXT PREPROCESSING")
print("="*50)

def preprocess_text(text):
    """
    Text ko clean karo:
    1. Lowercase karo
    2. Special characters hatao
    3. Extra spaces hatao
    """
    # Lowercase
    text = str(text).lower()
    
    # Alphanumeric aur spaces sirf rakhna
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Extra spaces hatao
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)

print("✅ Text preprocessing complete!")
print("\nOriginal vs Cleaned:")
for i in range(3):
    print(f"\nOriginal: {df['message'].iloc[i][:50]}...")
    print(f"Cleaned:  {df['cleaned_message'].iloc[i][:50]}...")

# ============================================
# Step 4: TEXT VECTORIZATION (TF-IDF)
# ============================================
print("\n" + "="*50)
print("TEXT VECTORIZATION (TF-IDF)")
print("="*50)

"""
TF-IDF = Term Frequency - Inverse Document Frequency
Yaani: Kaun se words spam/ham me important hain?
"""

vectorizer = TfidfVectorizer(
    max_features=5000,  # Top 5000 words use karo
    stop_words='english',  # "the", "a" etc hatao
    ngram_range=(1, 2)  # Single + double words
)

# Features banao
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

print(f"✅ Vectorization complete!")
print(f"Feature matrix shape: {X.shape}")
print(f"  → {X.shape[0]} messages")
print(f"  → {X.shape[1]} features (words/word-pairs)")

# ============================================
# Step 5: TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Data split ho gaya!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ============================================
# Step 6: NAIVE BAYES MODEL TRAIN KARO
# ============================================
print("\n" + "="*50)
print("NAIVE BAYES MODEL TRAINING")
print("="*50)

"""
Naive Bayes: Probability-based classifier
Yaani: Probability calculate karega ki message spam hai ya nahi
"""

print("\n🤖 Naive Bayes training ho raha hai...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("✅ Model trained!")

# ============================================
# Step 7: PREDICTIONS KARO
# ============================================
print("\n" + "="*50)
print("PREDICTIONS")
print("="*50)

y_pred = nb_model.predict(X_test)
y_pred_proba = nb_model.predict_proba(X_test)

print("✅ Predictions complete!")
print(f"\nPredictions (first 10):")
print(y_pred[:10])

# ============================================
# Step 8: EVALUATION METRICS
# ============================================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\n" + "="*30)
print("Metrics ka matlab:")
print("="*30)
print("Accuracy:  Overall kitne sahi predict kiye?")
print("Precision: Spam bataya jo vo sach me spam tha?")
print("Recall:    Total spam me se kitne pakde?")
print("F1-Score:  Precision + Recall ka balance")

# ============================================
# Step 9: CONFUSION MATRIX
# ============================================
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"True Negatives (Sahi HAM detect): {cm[0,0]}")
print(f"False Positives (Galat SPAM kaha): {cm[0,1]}")
print(f"False Negatives (Galat HAM kaha): {cm[1,0]}")
print(f"True Positives (Sahi SPAM detect): {cm[1,1]}")

# ============================================
# Step 10: VISUALIZATION
# ============================================
print("\n" + "="*50)
print("GRAPHS BANATE HAIN")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_xticklabels(['HAM (0)', 'SPAM (1)'])
axes[0, 0].set_yticklabels(['HAM (0)', 'SPAM (1)'])

# Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
axes[0, 1].bar(metrics_names, metrics_values, color=colors)
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Score')
for i, v in enumerate(metrics_values):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Spam vs Ham distribution
spam_ham = df['label'].value_counts()
axes[1, 0].pie(spam_ham, labels=['HAM', 'SPAM'], autopct='%1.1f%%', 
               colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1, 0].set_title('Dataset Distribution', fontsize=14, fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Graphs ready!")

# ============================================
# Step 11: SAMPLE SPAM/HAM PREDICTION
# ============================================
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

test_messages = [
    "Congratulations! You have won £1000 CLICK HERE NOW!!!",
    "Hey, how are you? Let's catch up soon!",
    "URGENT: Your account is compromised. Click to fix.",
    "Meeting at 3 PM tomorrow. Please confirm.",
]

for msg in test_messages:
    cleaned = preprocess_text(msg)
    msg_vectorized = vectorizer.transform([cleaned])
    prediction = nb_model.predict(msg_vectorized)[0]
    probability = nb_model.predict_proba(msg_vectorized)[0]
    
    label = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    
    print(f"\nMessage: {msg[:60]}...")
    print(f"Prediction: {label}")
    print(f"  → HAM confidence: {probability[0]:.2%}")
    print(f"  → SPAM confidence: {probability[1]:.2%}")

# ============================================
# Step 12: DETAILED CLASSIFICATION REPORT
# ============================================
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)

print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))

# ============================================
# Step 13: TOP SPAM INDICATORS
# ============================================
print("\n" + "="*50)
print("TOP WORDS INDICATING SPAM")
print("="*50)

feature_names = vectorizer.get_feature_names_out()
spam_class_prob = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
top_spam_indices = np.argsort(spam_class_prob)[-10:]

print("\nTop 10 words indicating SPAM:")
for idx in reversed(top_spam_indices):
    print(f"  • {feature_names[idx]}")

print("\n" + "="*50)
print("✅ PROJECT 2 COMPLETE!")
print("="*50)
