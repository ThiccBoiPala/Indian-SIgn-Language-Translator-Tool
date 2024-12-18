import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check the distribution of labels
label_counts = Counter(labels)
print("Label distribution before filtering:", label_counts)

# Filter out classes with fewer than 2 samples
filtered_data = []
filtered_labels = []
for label, count in label_counts.items():
    if count >= 2:
        indices = np.where(labels == label)[0]
        filtered_data.extend(data[indices])
        filtered_labels.extend(labels[indices])

filtered_data = np.asarray(filtered_data)
filtered_labels = np.asarray(filtered_labels)

# Check the distribution of labels after filtering
label_counts = Counter(filtered_labels)
print("Label distribution after filtering:", label_counts)

x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.4, shuffle=True, stratify=filtered_labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)