#!/usr/bin/env python3

import argparse
import sys
import argparse,struct

def rol32(x,k):
    """Auxiliary function (left rotation for 32-bit words)"""
    return ((x << k) | (x >> (32-k))) & 0xffffffff

def murmur3_32(key, seed):
    """Computes the 32-bit murmur3 hash"""
    # use the implementation from Problem 1
    k1 = 0xcc9e2d51
    k2 = 0x1b873593
    hash1 = seed & 0xffffffff
    key = bytearray(key.encode('utf-8'))
    length_key = len(key)
    number_of_blocks = length_key // 4

    for start_block in range(0, number_of_blocks * 4, 4):
        key1 = struct.unpack_from('<I', key, start_block)[0]
        key1 = (key1 * k1) & 0xffffffff
        key1 = rol32(key1, 15)
        key1 = (key1 * k2) & 0xffffffff

        hash1 ^= key1
        hash1 = rol32(hash1, 13)

        hash1 = (hash1 * 5 + 0xe6546b64) & 0xffffffff

    tail = key[number_of_blocks * 4:]
    key1 = 0
    if len(tail) >= 3:
        key1 ^= tail[2] << 16

    if len(tail) >= 2:
        key1 ^= tail[1] << 8
    if len(tail) >= 1:
        key1 ^= tail[0]
        key1 = (key1 * k1) & 0xffffffff
        key1 = rol32(key1, 15)
        key1 = (key1 * k2) & 0xffffffff
        hash1 ^= key1

    # Finalization
    hash1 ^= length_key
    hash1 ^= (hash1 >> 16)
    hash1 = (hash1 * 0x85ebca6b) & 0xffffffff
    hash1 ^= (hash1 >> 13)
    hash1 = (hash1 * 0xc2b2ae35) & 0xffffffff
    hash1 ^= (hash1 >> 16)

    return hash1

def auto_int(x):
    """Auxiliary function to help convert e.g. hex integers"""
    return int(x,0)

def dlog2(n):
    """Auxiliary function to compute discrete base2 logarithm"""
    print(f"dlog2 {n}")
    return n.bit_length() - 1

def rho(n):
    """Given a 32-bit number n, return the 1-based position of the first
    1-bit"""
    print(f"hex {h:08x}")
    if n==0:
        return 0
    binary_str = bin(n)[2:].zfill(32) 
    binary_list = list(binary_str)
    print(binary_list)
    for i in range(len(binary_list)):
        if binary_list[i] == '1':
            return i+1


def compute_jr(key,seed,log2m):
    """hash the string key with murmur3_32, using the given seed
    then take the **least significant** log2(m) bits as j
    then compute the rho value **from the left**

    E.g., if m = 1024 and we compute hash value 0x70ffec73
    or 0b01110000111111111110110001110011
    then j = 0b0001110011 = 115
         r = 2
         since the 2nd digit of 0111000011111111111011 is the first 1

    Return a tuple (j,r) of integers
    """
    h = murmur3_32(key,seed)     
    print(f'h {h:08x}')
    j = ~(0xffffffff << log2m) & h    
    r = rho(h)
    return j, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes (j,r) pairs for input integers.'
    )
    parser.add_argument('key',nargs='*',help='key(s) to be hashed',type=str)
    parser.add_argument('-s','--seed',type=auto_int,default=0,help='seed value')
    parser.add_argument('-m','--num-registers',type=int,default=2**58,
                            help=('Number of registers (must be a power of two)'))
    args = parser.parse_args()

    seed = args.seed
    m = args.num_registers
    if m <= 0 or (m&(m-1)) != 0:
        sys.stderr.write(f'{sys.argv[0]}: m must be a positive power of 2\n')
        quit(1)

    log2m = dlog2(m)

    # keys = []
    # with open(args.key[0], 'r', encoding='utf-8') as f:
    #     for line in f:
    #         key = line.strip()
    #         if key:
    #             keys.append(key)

    for key in args.key:
        h = murmur3_32(key,seed)

        j, r = compute_jr(key,seed,log2m)

        print(f'{key}\t{j}\t{r}')
        