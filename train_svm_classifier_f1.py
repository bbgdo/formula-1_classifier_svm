import pickle
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_time = time.time()

categories = ['Mercedes', 'Red Bull']
IMG_SIZE = 256

data = pickle.load(open('f1_data.pickle', 'rb'))

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

# Training
xfeatures, xtest, ylabels, ytest = train_test_split(features, labels, test_size=0.2, shuffle=True)

model = SVC(C=1, gamma='auto', kernel='poly', probability=True)
model.fit(xfeatures, ylabels)

predictions = model.predict(xtest)
accuracy = accuracy_score(ytest, predictions)

# Model dump
pickle.dump((model, accuracy), open('f1_svm.sav', 'wb'))

# Metrics
unique_labels = list(set(ytest))
cm = confusion_matrix(ytest, predictions, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix.png")
plt.close()

report = classification_report(ytest, predictions, target_names=categories, output_dict=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues', ax=ax)
plt.title("Classification Report")
plt.savefig("classification_report.png")
plt.close()

end_time = time.time()
total_time = end_time - start_time
print(f"Execution time: {total_time:.2f} seconds")
print(f"Model accuracy: {accuracy:.2%}")

