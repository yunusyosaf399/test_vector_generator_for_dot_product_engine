import numpy as np
import struct

def int8_to_hex_packed(arr):
    """Pack 64 int8 values into a 512-bit hex string (big-endian)"""
    assert len(arr) == 64
    return ''.join([format(x & 0xFF, '02x') for x in arr[::-1]])  # MSB first

def fp32_to_hex(val):
    """Convert a float32 to its IEEE 754 hex representation"""
    return format(struct.unpack('<I', struct.pack('<f', val))[0], '08x')

def generate_test_vectors(num_vectors=4, seed=42):
    np.random.seed(seed)
    test_vectors = []

    for i in range(num_vectors):
        A_int8 = np.random.randint(-128, 127, size=64, dtype=np.int8)
        x_fp32 = np.random.uniform(-1.0, 1.0, size=64).astype(np.float32)

        dot = np.dot(A_int8.astype(np.float32), x_fp32)
        a_hex = int8_to_hex_packed(A_int8)
        x_hex = [fp32_to_hex(f) for f in x_fp32]
        dot_hex = fp32_to_hex(dot)

        test_vectors.append({
            "A_row_packed": a_hex,
            "x_vector_fp32": x_hex,
            "expected_fp32_result": dot_hex
        })

    return test_vectors

def save_test_vectors(filename="test_vectors.mem", vectors=None):
    with open(filename, "w") as f:
        for v in vectors:
            f.write(f"# A_row_packed:\n{v['A_row_packed']}\n")
            f.write(f"# x_vector_fp32 (64 values):\n")
            f.write(" ".join(v['x_vector_fp32']) + "\n")
            f.write(f"# Expected dot product (fp32 hex):\n{v['expected_fp32_result']}\n")
            f.write("\n")

if __name__ == "__main__":
    vecs = generate_test_vectors(num_vectors=4)
    save_test_vectors("test_vectors.mem", vecs)
