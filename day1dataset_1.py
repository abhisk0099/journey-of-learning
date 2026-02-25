import numpy as np

# 1️⃣ Create Dataset (Area, Bedrooms, Price)
data = np.array([
    [1200, 2, 200000],
    [1500, 3, 250000],
    [1800, 4, 300000],
    [2000, 4, 350000],
    [2200, 5, 400000]
])

print("Full Dataset:\n", data)

# 2️⃣ Shape of dataset
print("\nShape:", data.shape)

# 3️⃣ Separate Features (X) and Target (y)
X = data[:, :2]   # First 2 columns
y = data[:, 2]    # Last column

print("\nFeatures (X):\n", X)
print("\nTarget (y):\n", y)

# 4️⃣ Mean Area and Bedrooms
print("\nMean of Features:", np.mean(X, axis=0))

# 5️⃣ Maximum price
print("\nMaximum Price:", np.max(y))

# 6️⃣ Reshape y for ML model
y = y.reshape(-1, 1)
print("\nReshaped y:\n", y)

# 7️⃣ Normalize Features (Important in ML)
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

print("\nNormalized Features:\n", X_normalized)