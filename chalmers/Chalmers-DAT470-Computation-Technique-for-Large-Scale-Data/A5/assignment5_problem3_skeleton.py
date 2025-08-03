#!/usr/bin/env python3

import datetime
import random
import argparse,struct
import sys
import os
from pyspark import SparkContext, SparkConf
import math
import matplotlib.pyplot as plt
import time
from utils import get_random_number, write_output, plot_summarised_data

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
    return n.bit_length() - 1

def rho(n):
    # Copy from Problem 2
    shifted = n >> log2m
    if shifted == 0:
        return 0
    return (32-log2m) - shifted.bit_length() + 1

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
    failed_keys = []
    error_messages = []
    # Copy from Problem 2
    try:
        h = murmur3_32(key,seed)
        #print(f'{h:08x}')
        j = ~(0xffffffff << log2m) & h
        r = rho(h)
    except Exception as e:
        error_messages.append(e)
        failed_keys.append(key)
        #print(f"Key {key} failed")
    sample_len = min(len(failed_keys), 10)
    if sample_len > 0:
        print(f"Total Failed keys : {len(failed_keys)}, \n Sample : {random.sample(failed_keys, sample_len)} \n error_samples: {random.sample(error_messages, sample_len)}")
    

def get_files(path):
    """
    A generator function: Iterates through all .txt files in the path and
    returns the content of the files

    Parameters:
    - path : string, path to walk through

    Yields:
    The content of the files as strings
    """
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                path = f'{root}/{file}'
                with open(path,'r') as f:
                    yield f.read()

def alpha(m):
    """Auxiliary function: bias correction"""
    if m ==16:
        return 0.673
    elif m ==32:
        return 0.697
    elif m == 64: 
        return 0.709
    elif m >= 128:
        return 0.7213 / (1+1.079/m)
    else:
        return ValueError("This value of alpha is not supported")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate the number of distinct words using HyperLogLog.'
    )
    parser.add_argument('path', help='path to directory', type=str)
    parser.add_argument('-s', '--seed', type=auto_int, default=None, help='seed value')
    parser.add_argument('-m', '--num-registers', type=int, required=True, help='number of registers (power of 2)')
    parser.add_argument('-w', '--num-workers', type=int, default=1, help='number of Spark workers')
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = get_random_number()
    m = args.num_registers
    if m <= 0 or (m & (m - 1)) != 0:
        sys.stderr.write('m must be a positive power of 2\n')
        quit(1)

    log2m = dlog2(m)

    num_workers = args.num_workers
    if num_workers < 1:
        sys.stderr.write('num-workers must be positive\n')
        quit(1)

    path = args.path
    if not os.path.isdir(path):
        sys.stderr.write(f'{path} is not a valid directory\n')
        quit(1)

    start = time.time()
    conf = SparkConf().setMaster(f'local[{num_workers}]').setAppName("HyperLogLog")
    conf.set('spark.driver.memory', '4g')
    sc = SparkContext(conf=conf)

    data = sc.parallelize(get_files(path))

    words = data.flatMap(lambda text: text.split()).cache()
    # Compute (j, r) pairs and find the max r per register j
    registers = (words
                 .map(lambda word: compute_jr(word, seed, log2m))
                 .filter(lambda pair: pair is not None)
                 .reduceByKey(lambda a, b: max(a, b))
                 .collect()
                 )

    M = [0] * m
    for j, r in registers:
        M[j] = r

    Z = 1.0 / sum([2.0 ** -r for r in M])
    alpha_m = alpha(m)
    E = alpha_m * m * m * Z

    # Correction for small range
    V = M.count(0)
    if E <= 2.5 * m:
        if V != 0:
            E = m * math.log(m / V)

    end = time.time()

    output  = {
        "num_workers": num_workers,
        "seed": seed,
        "num_registers": m,
        "start": start,
        "end": end,
        "time_taken": f"{end - start:.2f}",
        "path": str(path)
    }

    # print(f'Cardinality estimate: {round(E)}')
    print(f'Number of workers: {num_workers}')
    print(f'Took {end - start:.2f} s')

    prefix = f"{path.split('/')[-1]}-workers-{num_workers}-"
    write_output(output, print_data=True, file_prefix=prefix, child_dir=str(datetime.datetime.now()))