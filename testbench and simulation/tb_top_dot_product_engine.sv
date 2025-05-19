module tb_top_dot_product_engine;

  logic clk = 0;
  logic rst = 1;
  logic rstn = 0;

  logic en = 1;
  logic start;
  logic [15:0] weight_element;
  logic [2:0]  data_type;
  logic [3:0]  x_wr_addr;
  logic [31:0] x_wr_data;
  logic [63:0] x_wr_en;

  logic [31:0] y_out;
  logic        done;

  int i;

  // Test vector memories
  logic [15:0] a_mem [0:63];     // Packed A vector (can be 32 for int8)
  
  // Clock generation (200 MHz)
  always #2.5 clk = ~clk;

  // DUT instantiation
  top_dot_product_engine dut (
    .clk           (clk),
    .rst           (rst),
    .rstn          (rstn),
    .en            (en),
    .start         (start),
    .weight_element(weight_element),
    .data_type     (data_type),
    .x_wr_addr     (x_wr_addr),
    .x_wr_data     (x_wr_data),
    .x_wr_en       (x_wr_en),
    .y_out         (y_out),
    .done          (done)
  );

  // Function to interpret 32-bit logic as real
  function real fp32_to_real(input logic [31:0] val);
    int unsigned temp;
    begin
      temp = val;
      fp32_to_real = $bitstoshortreal(temp);
    end
  endfunction

  // Task to preload x_vector[i] = 1.0 (FP32)
  task preload_bram;
    for (i = 0; i < 64; i++) begin
      @(posedge clk);
      x_wr_en   = 64'b1 << i;
      x_wr_data = 32'h3F800000; // 1.0 in FP32
      x_wr_addr = 4'd0;
    end
    @(posedge clk);
    x_wr_en = 64'd0;
  endtask

  initial begin
    // Initialization
    start = 0;
    weight_element = 0;
    data_type = 3'd2; // INT8
    x_wr_en = 0;
    x_wr_data = 0;
    x_wr_addr = 0;

    // Reset sequence
    #20;
    rst = 0;
    rstn = 1;

    // Load test vectors (A_row_packed.mem)
    $readmemh("A_row_packed.mem", a_mem);

    // Preload x_vector (same as old tb)
    preload_bram();

    // Feed weight_element with loaded A values
    for (i = 0; i < 32; i++) begin
      @(posedge clk);
      weight_element = a_mem[i];
    end

    // Trigger start
    @(posedge clk); start = 1;
    @(posedge clk); start = 0;

    // Wait for result
    wait(done);
    $display("✅ Dot product result = %h (%f)", y_out, fp32_to_real(y_out));

    #20;
    $finish;
  end

endmodule