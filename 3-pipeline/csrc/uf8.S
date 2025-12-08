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
    addi sp, sp, -16              
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    sw s2, 0(sp)
    mv s0, a0                        # s0 = a0 = value
    li t0, 16                        # value = 16
    bltu s0, t0, encode_return_value # if value < 16, return value
    # Calculate exponent = 31 - clz((value >> 4) + 1)
    srli a0, s0, 4                   # value >> 4
    addi a0, a0, 1                   # (value >> 4) + 1
    jal ra, clz                      # call clz
    li t0, 31
    sub s1, t0, a0                   # exponent = 31 - clz
    # if (exponent > 15) exponent = 15
    li t0, 15
    bge t0, s1, calc_overflow
    mv s1, t0                        # s1 = exponent
calc_overflow:
    # overflow = ((1u << exponent) - 1u) << 4   
    li   t0, 1
    sll  t0, t0, s1                  # t0 = 1 << exponent
    addi t0, t0, -1                  # t0 = (1 << exponent) - 1
    slli s2, t0, 4                   # s2 = overflow = ((1 << exponent) - 1) << 4
calc_mantissa:
    # mantissa = (value - overflow) >> exponent
    sub t0, s0, s2                   # t0 = mantissa = value - overflow
    srl t0, t0, s1                   # mantissa >> exp
    slli s1, s1, 4                   # exp << 4
    or a0, s1, t0                    # return (mantissa | (exp << 4))
    j encode_end
encode_return_value:
    mv a0, s0                        # return value (a0)
encode_end:
    lw ra, 12(sp)             
    lw s0, 8(sp)             
    lw s1, 4(sp)             
    lw s2, 0(sp)              
    addi sp, sp, 16             
    ret