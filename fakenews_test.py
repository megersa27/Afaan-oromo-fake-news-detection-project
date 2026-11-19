import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Daataa Fakkeenyaa Afaan Oromoo (12 samples - 6 fake, 6 real)
data = {
    'news': [
        'Mootummaan qarshii kuma 500 tokkoon tokkoon lammiif laachuuf murteesse jedhamee odeeffamaa jira',
        'Yuunivarsiitiin Saayinsii fi Teeknooloojii Adaamaa sagantaa leenjii barattootaaf qopheesse',
        'Qorichi dhukkuba hundaa daqiiqaa shan keessatti fayyisu magaalaa keessatti gurguramaa jira jedhame',
        'Waajjirri Eegumsa Fayyaa talaallii loltootaa fi daa’immaniif kennamu ilaalchisee ibsa baase',
        'Namni tokko bishaan dhuguun malee waggaa kudhan jiraate jedhamee miidiyaa hawaasaa irratti tamsa’e',
        'Walga’iin misooma qonnaa naannoo Oromiyaa gaggeeffamu irratti ogeessonni hirmaatan',
        'Baankiin guddaan maallaqa namoota hundaaf dachaa godhee deebisuuf jedha odeeffannoon hin mirkanoofne',
        'Dorgommiin atleetiksii biyyoolessaa torban dhufu magaalaa Finfinneetti ni eegala',
        'Bilbila qaalii tolaan kennuuf jedha linkii kana cuqaasaa ergaa jedhuun namoonni gowwoomfamaa jiru',
        'Barsiisonni fi barattoonni qophii qormaataa xumuruuf sagantaa dabalataa jalqaban',
        'Marsaa ji’aa keessatti namni hundi kuma dhibba tokko argata oduu jedhu sobaadha',
        'Manni maree naannoo Oromiyaa bajata waggaa haaraa irratti mari’atee raggaasise'
    ],
    'label': [
        'fake', 'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake', 'real'
    ]
}

df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'fake': 1, 'real': 0})

X = df['news']
y = df['label_num']

# 2. Train/Test split (stratify=y gosa fake fi real walqixa qooduuf gargaara)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. TF-IDF Vectorization (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Modelii Leenjisuu
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)
print("\nModeliin irra deebi'amee leenjifiameera!")

# 5. Madaallii
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Bu'aa Madaallii Modelichaa ---")
print(f"Accuracy (Sirrummaa): {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['real', 'fake'], zero_division=0))

# 6. Oduu haaraa tilmaamuu
new_news = [
    "Maallaqa bilisaa account keessan irratti argachuuf as tuqaa jedhamee ergaan sobaa ergamaa jira",
    "Barattoonni kutaa 12ffaa qormaata biyyoolessaa qophaa'aa jiru jedhee ministeerri barnootaa beeksise"
]

new_news_vec = vectorizer.transform(new_news)
predictions = model.predict(new_news_vec)
probabilities = model.predict_proba(new_news_vec)

print("\n--- Tilmaama Oduu Haaraa ---")
for item, pred, probs in zip(new_news, predictions, probabilities):
    result = "FAKE NEWS 🚨 (Oduu Sobaa)" if pred == 1 else "REAL NEWS ✅ (Oduu Dhugaa)"
    conf = probs[pred] * 100
    print(f"Oduu: '{item}'")
    print(f"Bu'aa: {result} (Confidence: {conf:.1f}%)\n")