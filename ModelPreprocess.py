import pandas as pd
import seaborn as sns

# Load cleaned Titanic data
df = sns.load_dataset('titanic')

# Data Preprocessing
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df = df.drop('deck', axis=1)
df = df.drop_duplicates()
df['family_size'] = df['sibsp'] + df['parch'] + 1

df.info()

print(df.head())




df_encoded = pd.get_dummies(df, columns = ['sex','embarked'], drop_first = True)


df_encoded = pd.get_dummies(df_encoded, columns=['class'], drop_first=True)


print(df_encoded)


# Feature Standardization
from sklearn.preprocessing import StandardScaler, PowerTransformer

scaler = StandardScaler()

df_encoded[["age","family_size"]] = scaler.fit_transform(df_encoded[["age","family_size"]])

pt = PowerTransformer(method = 'yeo-johnson')
df_encoded['fare'] = pt.fit_transform(df_encoded[["fare"]])



print(df_encoded.head(5))




# Feature Selection for Model Training

drop_col = ['survived','who','embark_town','alive','adult_male','alone']  # Multicolinear and target Features

drop_col = [col for col in drop_col if col in df_encoded.columns]
x = df_encoded.drop(drop_col,axis = 1)
y = df_encoded['survived']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42, stratify = y)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Classification Report:\n", classification_report(y_test, y_pred))



