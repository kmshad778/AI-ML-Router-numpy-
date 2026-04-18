import numpy as np
import pandas as pd

print("=" * 50)
print("PART 1: NumPy Operations")
print("=" * 50)

arr_arange = np.arange(1, 11)           
arr_linspace = np.linspace(0, 1, 5)    
arr_random = np.random.random((3, 3))   

print("\narange()  :", arr_arange)
print("linspace():", arr_linspace)
print("random()  :\n", arr_random)

print("\nShape    :", arr_random.shape)
print("Dimension:", arr_random.ndim)
print("Data Type:", arr_random.dtype)

data = np.array([23, 45, 12, 67, 34, 89, 10, 55])

print("\nArray    :", data)
print("Mean     :", np.mean(data))
print("Median   :", np.median(data))
print("Std Dev  :", np.std(data))
print("Min      :", np.min(data))
print("Max      :", np.max(data))

print("\n" + "=" * 50)
print("PART 2: Pandas Basic Operations")
print("=" * 50)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\n" + "=" * 50)
print("PART 3: Data Cleaning")
print("=" * 50)

print("\nMissing Values:")
print(df.isnull().sum())

df_dropped = df.dropna(subset=['Cabin'])
print("\nAfter dropna() on Cabin:", df_dropped.shape)

df['Age'] = df['Age'].fillna(df['Age'].mean())
print("After fillna() on Age - Missing Age values:", df['Age'].isnull().sum())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"Duplicates removed: {before - after} rows | Remaining rows: {after}")

print("\n" + "=" * 50)
print("PART 4: Data Manipulation")
print("=" * 50)

selected = df[['Name', 'Age', 'Sex', 'Pclass', 'Survived']]
print("\nSelected Columns:")
print(selected.head())

age_filter = df[df['Age'] > 30]
print(f"\nPassengers with Age > 30: {len(age_filter)}")
print(age_filter[['Name', 'Age', 'Sex']].head())

female = df[df['Sex'] == 'female']
print(f"\nFemale Passengers: {len(female)}")
print(female[['Name', 'Age', 'Sex']].head())

df['Family_Size'] = df['SibSp'] + df['Parch']
print("\nNew column 'Family_Size' added:")
print(df[['Name', 'SibSp', 'Parch', 'Family_Size']].head())

print("\n✅ All 4 Parts Completed Successfully!")
