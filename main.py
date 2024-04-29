# LWE inner-product implementation

import numpy as np
from collections import Counter

import numpy as np


def sorted_labels_and_indices(labels):
    # Find unique labels and sort them alphabetically
    unique_labels = np.unique(labels)
    sorted_labels = np.sort(unique_labels)

    # Create a list to hold the sorted labels and their indices
    result = []
    for label in sorted_labels:
        # Find the indices for each label
        indices = np.where(labels == label)[0]
        result.append((label, indices.tolist()))

    return result


# Load the .npy file
data_embeddings = np.load('LFW_embeddings.npy')
data_labels = np.load('LFW_labels.npy')

# Display basic information about the array
print(f"Embeddings")
print(f"Shape of the array: {data_embeddings.shape}")
print(f"Size of the array: {data_embeddings.size}")
print(f"Data type of the array: {data_embeddings.dtype}")

# If the data is numerical, display some basic statistics
if np.issubdtype(data_embeddings.dtype, np.number):
    print(f"Mean: {np.mean(data_embeddings)}")
    print(f"Median: {np.median(data_embeddings)}")
    print(f"Standard Deviation: {np.std(data_embeddings)}")
else:
    print("Data is not numerical.")
print("First element of the array:", data_embeddings[:1])

print(f"-----------------------------")
# Display basic information about the labels
print(f"Labels")
print(f"Shape of the array: {data_labels.shape}")
print(f"Size of the array: {data_labels.size}")
print(f"Data type of the array: {data_labels.dtype}")

# Get sorted labels and their indices
sorted_labels_indices = sorted_labels_and_indices(data_labels)

# Print the result
for label, indices in sorted_labels_indices:
    print(f"Label: {label}, Indices: {indices}")

# # Print labels count
# label_counts = Counter(data_labels)
# more_than_one = 0
# for label, count in label_counts.most_common():
#     if count > 1:
#         more_than_one = more_than_one + 1
#     print(f"{label}: {count}")
# print(f"{more_than_one}")

# print(f"First label: {data_labels[0]}")
# print(f"Second label: {data_labels[1]}")
# print(f"Third label: {data_labels[2]}")
# Label: Colin_Powell, Indices: [0, 133, 156, 172, 227, 359, 393, 396, 422, 437, ...]