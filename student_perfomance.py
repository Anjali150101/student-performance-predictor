import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load CSV file
df = pd.read_csv("student_data.csv")

# Encode target column
le = LabelEncoder()
df['performance'] = le.fit_transform(df['performance'])  # pass=1, fail=0

# Separate features and label
X = df.drop('performance', axis=1)
y = df['performance']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Predict with valid feature names
sample = pd.DataFrame([[1,25,30,5]], columns=(["hours_studied","attendance","previous_score", "sleep_hours",]))
prediction = model.predict(sample)

print("\nPrediction for sample student:",'pass' if prediction[0] == 1 else 'Fail'
      )#prediction[0])
