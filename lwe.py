import numpy as np


def setup(low, row, col):
    return np.random.randint(low, size=(row, col))


def encode(u, b):

    if b == 0:
        rb = setup(q, m, (m - k))  # see section 2.1, dimension is m x (m-k)
        # Compute H.T * r0
        h_rb = np.dot(H.T, rb)

        # Check if vector u needs to be concatenated with zeros to allow subtraction of H.T * rb
        required_columns = h_rb.shape[1] - u.shape[1]
        if required_columns > 0:
            zero_padding = np.zeros((u.shape[0], required_columns))
            u_padded = np.hstack((u, zero_padding))
        else:
            u_padded = u

        # Compute (q/p) * u
        scaled_u = (q / p) * u_padded

        # Compute the final result
        pkb = scaled_u - h_rb

        return pkb, rb

    if b == 1:
        rb = setup(q, m, (m - k))  # see section 2.1, dimension is m x (m-k)
        # Check if vector u needs to be concatenated with samples s to match the size for multiplication with H
        required_rows = H.shape[1]
        if required_rows > 0:
            skb = setup(q, required_rows, rb.shape[1])
            skb[0, :u.shape[1]] = u
        else:
            skb = u

        # Compute H dot u_padded
        scaled_h = np.dot(H, skb)

        # Compute final result
        pkb = scaled_h + rb

        return pkb, skb


def decode(pk_decode, sk_decode):
    # Matrix multiplication
    product = np.dot(pk_decode.T, sk_decode)

    # Multiply by p/q
    scaled_product = (p / q) * product

    # Apply round function element-wise, see lemma 5
    rounded_result = np.round(scaled_product)

    # Apply modulo q to each element of the rounded result
    final_result = np.mod(rounded_result, q)

    return final_result


# Parameters
n = 128  # Dimension of vectors
k = 3*n    # see section 3.2 + 3.5
m = 4*n  # see section 3.2 + 3.5
q = 3329
p = 15   # field doing inner products over
e = 5    # error rate (|e|+1)*(p/q) = negligible -> m*B^2*p/q < 2^-40

# Data
data_embeddings = np.load('LFW_embeddings.npy')
data_labels = np.load('LFW_labels.npy')

# Setup phase
H = setup(q, m, (k+n))  # Generator matrix

# Get two values, Colin Powell at indices 0 and 133
x = data_embeddings[:1]
print(f"{data_labels[0]}: {x}")
y = data_embeddings[133:134]
print(f"{data_labels[133]}: {y}")

# Encode phase
pk_x_0, sk_x_0 = encode(x, 0)
pk_x_1, sk_x_1 = encode(x, 1)
pk_y_0, sk_y_0 = encode(y, 0)
pk_y_1, sk_y_1 = encode(y, 1)

# Decode phase
z = decode(pk_x_0, sk_y_1)
z_prime = decode(pk_y_1, sk_x_0)

# Variant 1 Validation primitive according to page 2
product_x_y = np.dot(x.T, y)
sum_z_z_prime = z + z_prime
print(f"Product x.T * y: {product_x_y}")
print(f"Sum z + z': {sum_z_z_prime}")


# Variant 2 Validation of correctness according to page 9
left_side = np.dot(pk_x_0.T, sk_y_1) + np.dot(pk_y_1.T, sk_x_0)
temp_a = ((q/p) * np.dot(x.T, y))

r0 = setup(q, m, (m - k))  # see section 2.1, dimension is m x (m-k)
r1 = setup(q, m, (m - k))  # see section 2.1, dimension is m x (m-k)
temp_b = np.dot(r1.T, r0)
right_side = temp_a + temp_b
delta = left_side - right_side
delta_abs = np.abs(delta)
print(f"Delta Error: {delta_abs}")

# Page 17 in second paper, Hamming Distance for transformation of vector
# Ask Florias why, Hamming Distance gives the number of different values in two vectors. Here it would be 128 because the values are floating poing.