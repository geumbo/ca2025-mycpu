.text

.globl uf8_decode
.type uf8_decode,%function

.globl uf8_encode
.type uf8_encode,%function

clz:
    # if (x == 0) return 32
    bne  a0, x0, clz_start
    li   a0, 32
    ret                      
clz_start:
    li   a1, 0               # a1 = n = 0
    srli t0, a0, 16
    bne  t0, x0, check_8 
    addi a1, a1, 16          # n += 16
    slli a0, a0, 16          # x <<= 16
check_8:
    srli t0, a0, 24
    bne  t0, x0, check_4
    addi a1, a1, 8           # n += 8
    slli a0, a0, 8           # x <<= 8
check_4:
    srli t0, a0, 28
    bne  t0, x0, check_2
    addi a1, a1, 4           # n += 4
    slli a0, a0, 4           # x <<= 4
check_2:
    srli t0, a0, 30
    bne  t0, x0, check_1
    addi a1, a1, 2           # n += 2
    slli a0, a0, 2           # x <<= 2
check_1:
    blt a0, x0, clz_end      # if (x < 0), MSB is 1, we are done
    addi a1, a1, 1           # n += 1
clz_end:
    mv   a0, a1              # move result to a0 for return
    ret
uf8_decode:
    andi t0, a0, 0x0f   # t0 = mantissa = x & 0x0f
    srli t1, a0, 4      # t1 = exponent = x >> 4
    li t2, 15           # t2 = 15
    sub t3, t2, t1      # t3 = offset = 15 - exponent
    li t4, 0x7FFF       # t4 = 0x7FFF
    srl t3, t4, t3      # t3 = offset = 07FFF >> (15 - exponent)
    slli t3, t3, 4      # t3 = offset = 07FFF >> (15 - exponent) << 4
    sll  t0, t0, t1     # mantissa << exponent
    add a0, t0, t3      # return (mantissa << exponent) + offset
    ret
uf8_encode:
    # Optimized Leaf Function: No Stack Usage
    # Input: a0 (value)
    # Output: a0 (encoded)
    # Register Map:
    # a2: value
    # a3: exponent
    # a4: overflow
    mv a2, a0                        # n = value
    li t0, 16 
    bltu a2, t0, encode_return_value # if value < 16, return value
    # Calculate exponent = 31 - clz((value >> 4) + 1)
    srli a0, a2, 4                   # a0 = value >> 4
    addi a0, a0, 1                   # a0 = (value >> 4) + 1
    # --- INLINED CLZ START ---
    # Input: a0, Clobbers: a1, t0
    bne  a0, x0, opt_clz_start
    li   a0, 32
    j    opt_clz_done
opt_clz_start:
    li   a1, 0
    srli t0, a0, 16
    bne  t0, x0, opt_check_8 
    addi a1, a1, 16
    slli a0, a0, 16
opt_check_8:
    srli t0, a0, 24
    bne  t0, x0, opt_check_4
    addi a1, a1, 8
    slli a0, a0, 8
opt_check_4:
    srli t0, a0, 28
    bne  t0, x0, opt_check_2
    addi a1, a1, 4
    slli a0, a0, 4
opt_check_2:
    srli t0, a0, 30
    bne  t0, x0, opt_check_1
    addi a1, a1, 2
    slli a0, a0, 2
opt_check_1:
    blt a0, x0, opt_clz_end
    addi a1, a1, 1
opt_clz_end:
    mv   a0, a1
opt_clz_done:
    # --- INLINED CLZ END ---
    li t0, 31
    sub a3, t0, a0                   # a3 (exponent) = 31 - clz
    # if (exponent > 15) exponent = 15
    li t0, 15
    bge t0, a3, calc_overflow
    mv a3, t0                        # exponent = 15
calc_overflow:
    # overflow = ((1u << exponent) - 1u) << 4   
    li   t0, 1
    sll  t0, t0, a3                  # 1 << exponent
    addi t0, t0, -1                  # mask
    slli a4, t0, 4                   # a4 (overflow)
calc_mantissa:
    # mantissa = (value - overflow) >> exponent
    sub t0, a2, a4                   # value - overflow
    srl t0, t0, a3                   # >> exponent
    slli a3, a3, 4                   # exponent << 4
    or a0, a3, t0                    # return (mantissa | exp)
    ret
encode_return_value:
    mv a0, a2                        # return original value
    ret