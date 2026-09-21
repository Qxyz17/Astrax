max_cp = 0x10FFFF
for low_bits in (14, 10):
    high = (max_cp >> low_bits) + 1
    low = 1 << low_bits
    params = (high + low) * 192
    print(f'low_bits={low_bits}: high={high}, low={low}, params={params:,}')
