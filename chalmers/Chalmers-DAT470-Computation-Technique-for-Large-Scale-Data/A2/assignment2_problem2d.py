import os
import argparse
import sys
import time
import multiprocessing as mp
import matplotlib.pyplot as plt


def get_filenames(path):
    """
    A generator function: Iterates through all .txt files in the path and
    returns the full names of the files

    Parameters:
    - path : string, path to walk through

    Yields:
    The full filenames of all files ending in .txt
    """
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield f'{root}/{file}'

def get_file(path):
    """
    Reads the content of the file and returns it as a string.

    Parameters:
    - path : string, path to a file

    Return value:
    The content of the file in a string.
    """
    with open(path,'r') as f:
        return f.read()

def count_words_in_file(file):
    """
    Counts the number of occurrences of words in the file
    Whitespace is ignored

    Parameters:
    - file, string : the content of a file

    Returns: Dictionary that maps words (strings) to counts (ints)
    """
    counts = dict()
    for word in file.split():
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts



def get_top10(counts):
    """
    Determines the 10 words with the most occurrences.
    Ties can be solved arbitrarily.

    Parameters:
    - counts, dictionary : a mapping from words (str) to counts (int)
    
    Return value:
    A list of (count,word) pairs (int,str)
    """
    return sorted(((word_count,word) for word,word_count in counts.items()),reverse=True)[:10]



def merge_counts(dict_to, dict_from):
    """
    Merges the word counts from dict_from into dict_to, such that
    if the word exists in dict_to, then the count is added to it,
    otherwise a new entry is created with count from dict_from

    Parameters:
    - dict_to, dictionary : dictionary to merge to
    - dict_from, dictionary : dictionary to merge from

    Return value: None
    """
    for (k,v) in dict_from.items():
        if k not in dict_to:
            dict_to[k] = v
        else:
            dict_to[k] += v



def compute_checksum(counts):
    """
    Computes the checksum for the counts as follows:
    The checksum is the sum of products of the length of the word and its count

    Parameters:
    - counts, dictionary : word to count dictionary

    Return value:
    The checksum (int)
    """
    return sum(len(word) * word_count for word,word_count in counts.items())


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Counts words of all the text files in the given directory')
    parser.add_argument('-w', '--num-workers', help = 'Number of workers', default=1, type=int)
    parser.add_argument('-b', '--batch-size', help = 'Batch size', default=1, type=int)
    parser.add_argument('path', help = 'Path that contains text files')
    args = parser.parse_args()

    path = args.path

    if not os.path.isdir(path):
        sys.stderr.write(f'{sys.argv[0]}: ERROR: `{path}\' is not a valid directory!\n')
        quit(1)

    num_workers = args.num_workers
    if num_workers < 1:
        sys.stderr.write(f'{sys.argv[0]}: ERROR: Number of workers must be positive (got {num_workers})!\n')
        quit(1)

    batch_size = args.batch_size
    if batch_size < 1:
        sys.stderr.write(f'{sys.argv[0]}: ERROR: Batch size must be positive (got {batch_size})!\n')
        quit(1)

    start_reading=time.time()
    files = [get_file(fn) for fn in get_filenames(path)]
    finish_reading = time.time()
    total_read = finish_reading - start_reading

    time_to_count_words = time.time()
    file_counts = list()
    for file in files:
        file_counts.append(count_words_in_file(file))
    finish_counting_words = time.time()
    time_to_count = finish_counting_words - time_to_count_words
    global_counts = dict()


    file_counts = list()
    for file in files:
       file_counts.append(count_words_in_file(file))

    start_merging = time.time() 
    for counts in file_counts:
        merge_counts(global_counts,counts)
    end_merge_count = time.time()
    end_merge_count = time.time()
    time_to_merge = end_merge_count - start_merging

    start_parallelism = time.time()


    with mp.Pool(num_workers) as pool:
        pool.map(count_words_in_file,files)
    finish_parallelism = time.time()
    total_time = time.time() - start_time
    parallel_time = finish_parallelism - start_parallelism
    sequential_time = total_time - parallel_time
    upper_bound = 1/1-parallel_time
    # time output
    print("\n Time to:")
    print(f"Read data  : {total_read:.4f} seconds")
    print(f"count words  : {time_to_count_words:.4f} seconds")
    print(f"Merging count of words    : {time_to_merge:.4f} seconds")
    print(f"Total execution time   : {total_time:.4f} seconds")
    print(f"sequential-time : {sequential_time:.4f} seconds")
    print(f"Parallel time: {parallel_time:.4f}seconds")
    parallelizable_fraction = parallel_time/total_time
    upper_bound = 1/1-parallelizable_fraction
    print(f"\nFraction of parallelizable part is {parallelizable_fraction:.4f}\n")
    print(f"upper bound speedup: {upper_bound:.4f} seconds")

    # total count of words for .txt files in directory 
    #checksum=compute_checksum(global_counts)
    #print(f'Checksum : {checksum}')


    # get top 10 most common words from .txt file
    #ten_top=get_top10(global_counts)
    #for word_count, word in ten_top:
     #   print(f"The top words count are {word}:{word_count}")
