# Test Vectors Generator Script for Dot Product Engine (6 Data Types)

This Python script generates input test vectors and expected outputs for validating a hardware Dot Product Engine (DPE). It supports six different data types used in the pipeline and creates ready-to-use `.mem` files for Verilog/SystemVerilog simulation.

---

## ✅ Supported Data Types

You can select one of the following types:

| Option | Data Type |
|--------|------------|
| `1`    | `int2`     |
| `2`    | `int4`     |
| `3`    | `int8`     |
| `4`    | `uint8`    |
| `5`    | `fp16`     |
| `6`    | `bf16`     |

---

## 📁 Output Directory Structure

When you run the script and choose a type, it generates:

    test_vectors_by_type/
    ├── int4/
    │ ├── A_row_packed.mem # Input vector A
    │ ├── x_vector.mem # Input vector x
    │ └── expected_y.txt # Output dot product (float)
    ├── ...


Each directory corresponds to the data type selected.

---

## 📦 Files Explained

- `A_row_packed.mem`: Input vector A, formatted in 2's complement hex or IEEE-754 hex (for floats).
- `x_vector.mem`: 64 float32 numbers in IEEE-754 hex.
- `expected_y.txt`: Single output value representing the dot product result.

---

## 🚀 How to Use

1. Clone or download this repository.
2. Open terminal in the project directory.
3. Run the script:
   ```bash
   python generate_vectors.py


🔬 Example Use Case (Verilog Testbench)

Use $readmemh to load these files in your Verilog/SystemVerilog testbench:

    initial begin
    $readmemh("A_row_packed.mem", a_mem);
    $readmemh("x_vector.mem", x_mem);
    end


Compare y_out from simulation with the value in expected_y.txt.
📌 Requirements

Python 3.6+

Packages:

    numpy

Install with:

    pip install numpy

