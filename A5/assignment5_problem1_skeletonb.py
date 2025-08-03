#!/usr/bin/env python3
import datetime

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse,struct
import math
def rol32(x,k):
    """Auxiliary function (left rotation for 32-bit words)"""
    return ((x << k) | (x >> (32-k))) & 0xffffffff

def murmur3_32(key, seed):
    """Computes the 32-bit murmur3 hash"""
    k1 = 0xcc9e2d51
    k2 = 0x1b873593
    hash1 =seed & 0xffffffff
    key = bytearray(key.encode('utf-8'))
    length_key = len(key)
    number_of_blocks = length_key //4

    for start_block in range(0,number_of_blocks *4,4):
        key1 = struct.unpack_from('<I',key,start_block)[0]
        key1 = (key1 * k1) & 0xffffffff
        key1 = rol32(key1,15)
        key1 = (key1 * k2) & 0xffffffff

        hash1 ^=key1
        hash1 = rol32(hash1,13)

        hash1 =(hash1 * 5 + 0xe6546b64) & 0xffffffff

    tail = key[number_of_blocks *4:]
    key1 = 0
    if len(tail) >=3:
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

def evaluate_hash_distribution(hashes, m):
    mask = m - 1
    reduced_hashes = [h & mask for h in hashes]

    freq = Counter(reduced_hashes)
    values = list(freq.keys())
    counts = list(freq.values())

    plt.bar(values, counts)
    plt.xlabel('Hash Value (Least Significant 7 Bits)')
    plt.ylabel('Frequency')
    plt.title('Hash Value Distribution Histogram (Least Significant 7 Bits)')
    current_date = datetime.datetime.now()
    plt.savefig(f"{current_date}.png")

    mean_val = np.mean(reduced_hashes)
    std_val = np.std(reduced_hashes)
    return mean_val, std_val, freq

def evaluate_collisions(hashes, m):
    mask = m - 1
    reduced_hashes = [h & mask for h in hashes]
    freq = Counter(reduced_hashes)

    collisions = sum(math.comb(count, 2) for count in freq.values() if count > 1)
    total_pairs = math.comb(len(hashes), 2)
    collision_prob = collisions / total_pairs
    return collisions, total_pairs, collision_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes MurMurHash3 for the keys.'
    )
    parser.add_argument('key',nargs='*',help='key(s) to be hashed',type=str)
    parser.add_argument('-s','--seed',type=auto_int,default=0,help='seed value')
    parser.add_argument('-m', type=int, default=128)

    args = parser.parse_args()
    keys = []
    with open(args.key[0], 'r', encoding='utf-8') as f:
        for line in f:
            key = line.strip()
            if key:
                keys.append(key)

    seed = args.seed
    for key in keys:
        h = murmur3_32(key,seed)
        # print(f'{h:#010x}\t{key}')

    hashes = [murmur3_32(key, seed) for key in keys]

    mean_val, std_val, freq = evaluate_hash_distribution(hashes, args.m)
    print(f"Hash Value Distribution Evaluation (m={args.m}, using 7 bits):")
    print(f"Mean: {mean_val}")
    print(f"Standard Deviation: {std_val}")

    collisions, total_pair, collision_prob = evaluate_collisions(hashes, args.m)
    print(f"\nCollision Evaluation (m={args.m}, using 7 bits):")
    print(f"Number of Collisions: {collisions}")
    print(f"Total Pairs: {total_pair}")
    print(f"Collision Probability: {collision_prob}")

