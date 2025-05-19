import numpy as np
import struct

def pack_bits(arr, bits_per_elem):
    """
    Pack array of int values (positive/negative) into big-endian packed hex.
    Returns as a hex string (MSB first, as for Verilog memory initialization).
    """
    total_bits = len(arr) * bits_per_elem
    val = 0
    for i, x in enumerate(arr):
        val = (val << bits_per_elem) | (int(x) & ((1 << bits_per_elem) - 1))
    n_bytes = (total_bits + 7) // 8
    return int(val).to_bytes(n_bytes, byteorder="big").hex()


def fp32_to_hex(f):
    return format(struct.unpack('<I', struct.pack('<f', f))[0], '08x')

def fp16_to_hex(f):
    import ctypes
    # numpy float16 to uint16 bits
    return format(np.float16(f).view(np.uint16), '04x')

def bf16_to_hex(f):
    # bfloat16 is top 16 bits of fp32
    as_int = struct.unpack('<I', struct.pack('<f', f))[0]
    return format(as_int >> 16, '04x')

def rand_vec(data_type, size):
    if data_type == "int2":
        return np.random.randint(-2, 2, size=size, dtype=np.int8)
    if data_type == "int4":
        return np.random.randint(-8, 8, size=size, dtype=np.int8)
    if data_type == "int8":
        return np.random.randint(-128, 128, size=size, dtype=np.int8)
    if data_type == "uint8":
        return np.random.randint(0, 256, size=size, dtype=np.uint8)
    if data_type == "fp16" or data_type == "bf16":
        # -10.0 to +10.0 for variety
        return np.random.uniform(-10, 10, size=size).astype(np.float32)
    raise ValueError("Unknown data type")

def pack_row(row, data_type):
    if data_type == "int2":
        return pack_bits(row, 2)
    if data_type == "int4":
        return pack_bits(row, 4)
    if data_type == "int8":
        return pack_bits(row, 8)
    if data_type == "uint8":
        return pack_bits(row, 8)
    if data_type == "fp16":
        # Each as 2 bytes, big endian
        return ''.join([fp16_to_hex(x) for x in row])
    if data_type == "bf16":
        return ''.join([bf16_to_hex(x) for x in row])
    raise ValueError("Unknown data type")

def row_to_fp32(row, data_type):
    """Convert a row of any type to float32 numpy array."""
    if data_type in ["int2", "int4", "int8", "uint8"]:
        return row.astype(np.float32)
    if data_type == "fp16":
        return np.array(row, dtype=np.float16).astype(np.float32)
    if data_type == "bf16":
        # For demo: simulate by converting fp32 (real hardware will need proper unpack)
        return row.astype(np.float32)
    raise ValueError("Unknown data type")

def x_vector_hex(x_vector, data_type):
    # Returns a list of hex strings
    if data_type == "int2":
        return [format(int(x) & 0x3, '01x') for x in x_vector]
    if data_type == "int4":
        return [format(int(x) & 0xF, '01x') for x in x_vector]
    if data_type == "int8" or data_type == "uint8":
        return [format(int(x) & 0xFF, '02x') for x in x_vector]
    if data_type == "fp16":
        return [fp16_to_hex(x) for x in x_vector]
    if data_type == "bf16":
        return [bf16_to_hex(x) for x in x_vector]
    else:
        return [fp32_to_hex(float(x)) for x in x_vector]

def main(data_type="int8", n_rows=64, n_cols=64):
    np.random.seed(42)
    y_out = []
    with open("A_row_packed.mem", "w") as af, \
         open("x_vector.mem", "w") as xf, \
         open("y_out.txt", "w") as yf:
        # Generate and save X vector (all 1.0s)
        if data_type in ["int2", "int4", "int8", "uint8"]:
            xvec = np.ones(n_cols, dtype=np.int8 if "int" in data_type else np.uint8)
        else:
            xvec = np.ones(n_cols, dtype=np.float32)
        xf.write(' '.join(x_vector_hex(xvec, data_type)) + '\n')
        for row in range(n_rows):
            A = rand_vec(data_type, n_cols)
            # Pack this row and write to file
            packed = pack_row(A, data_type)
            af.write(packed + '\n')
            # Compute dot product as fp32
            Ad = row_to_fp32(A, data_type)
            Xd = row_to_fp32(xvec, data_type)
            dot = np.dot(Ad, Xd)
            yf.write(f"{fp32_to_hex(dot)}\n")
            print(f"Row {row}: Dot product = {dot} (fp32 hex: {fp32_to_hex(dot)})")
    print("All files written: A_row_packed.mem, x_vector.mem, y_out.txt")

if __name__ == "__main__":
    # User can set this value to any of: int2, int4, int8, uint8, fp16, bf16
    main(data_type="int8")
