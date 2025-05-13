import numpy as np
import struct
import os

def generate_vectors(data_type_choice):
    VECTOR_LENGTH = 64
    FLOAT_TYPE = np.float32

    # Define output directory structure
    base_dir = "./test_vectors_by_type"
    os.makedirs(base_dir, exist_ok=True)

    # Map type to data generation settings
    type_map = {
        1: {"name": "int2", "dtype": np.int8, "min": -2, "max": 3},
        2: {"name": "int4", "dtype": np.int8, "min": -8, "max": 8},
        3: {"name": "int8", "dtype": np.int8, "min": -128, "max": 128},
        4: {"name": "uint8", "dtype": np.uint8, "min": 0, "max": 256},
        5: {"name": "fp16", "dtype": np.float16, "min": -10.0, "max": 10.0},
        6: {"name": "bf16", "dtype": np.float32, "min": -10.0, "max": 10.0}
    }

    if data_type_choice not in type_map:
        raise ValueError("Invalid selection. Choose a number from 1 to 6.")

    dtype_info = type_map[data_type_choice]
    type_name = dtype_info["name"]
    out_dir = os.path.join(base_dir, type_name)
    os.makedirs(out_dir, exist_ok=True)

    # Generate input A vector
    if "int" in type_name or "uint" in type_name:
        A = np.random.randint(dtype_info["min"], dtype_info["max"], size=64, dtype=dtype_info["dtype"])
    else:
        A = np.random.uniform(dtype_info["min"], dtype_info["max"], size=64).astype(dtype_info["dtype"])

    # Generate x_vector as float32 values
    x_vector = np.random.uniform(-10.0, 10.0, size=64).astype(FLOAT_TYPE)

    # Compute dot product
    dot_product = np.dot(A.astype(FLOAT_TYPE), x_vector)

    # Save A to .mem file
    with open(os.path.join(out_dir, "A_row_packed.mem"), "w") as f:
        for val in A:
            if "int" in type_name or "uint" in type_name:
                f.write(f"{val & 0xFF:02X}\n")
            else:
                ieee_hex = struct.unpack('>I', struct.pack('>f', float(val)))[0]
                f.write(f"{ieee_hex:08X}\n")

    # Save x_vector
    with open(os.path.join(out_dir, "x_vector.mem"), "w") as f:
        for val in x_vector:
            ieee_hex = struct.unpack('>I', struct.pack('>f', val))[0]
            f.write(f"{ieee_hex:08X}\n")

    # Save expected result
    with open(os.path.join(out_dir, "expected_y.txt"), "w") as f:
        f.write(f"{dot_product:.6f}\n")

    print(f"✅ {type_name} test vectors generated in: {out_dir}")

# Main function with user input
def main():
    print("Select data type to generate test vectors:")
    print("1. int2\n2. int4\n3. int8\n4. uint8\n5. fp16\n6. bf16")
    try:
        input_type = int(input("Enter option (1-6): "))
        generate_vectors(input_type)
    except Exception as e:
        print(f"❌ Error: {e}")

# Run the main function
main()
