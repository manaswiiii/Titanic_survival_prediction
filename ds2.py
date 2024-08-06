# titanic-survival-prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf

# Load the Titanic dataset
titanic_df = pd.read_csv('C:/Users/manas/OneDrive/Desktop/everything/codsoft/data science/Titanic-Dataset.csv')

# Selecting features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Separate the features and target variable from the dataframe
X = titanic_df[features]
y = titanic_df[target]

# Preprocessing pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Applying the transformations
X_preprocessed = preprocessor.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Converting to tensorflow tensors
input_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
output_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
input_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
output_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Reshape the output to match the input shape of the neural network
output_train = tf.reshape(output_train, (-1, 1))
output_test = tf.reshape(output_test, (-1, 1))

# Creating a neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
print(model.summary())

# Training the model
model.fit(input_train, output_train, epochs=100, validation_data=(input_test, output_test))

# Example prediction
test_samples = [[3, 'male', 22, 1, 0, 7.25, 'S'],
                [1, 'female', 38, 1, 0, 71.2833, 'C']]
test_samples_preprocessed = preprocessor.transform(pd.DataFrame(test_samples, columns=features))
test_samples_tensor = tf.convert_to_tensor(test_samples_preprocessed, dtype=tf.float32)
predictions = model.predict(test_samples_tensor)
print(predictions)
