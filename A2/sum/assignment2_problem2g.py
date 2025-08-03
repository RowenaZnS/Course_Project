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

def count_words_in_file(filename_queue, wordcount_queue, batch_size):
    """
    Counts the number of occurrences of words in the file
    Performs counting until a None is encountered in the queue
    Counts are stored in wordcount_queue
    Whitespace is ignored

    Parameters:
    - filename_queue, multiprocessing queue: will contain filenames and None as a sentinel
    - wordcount_queue, multiprocessing queue: (word,count) dictionaries are put in the queue
    - batch_size, int: size of batches to process

    Returns: None
    """
    batch_counts = {}
    files_processed = 0
    
    while True:
        filename = filename_queue.get()
        
        # Check for end of input
        if filename is None:
            # Put any remaining counts before exiting
            if files_processed > 0:
                wordcount_queue.put(batch_counts)
            # Signal end of processing
            wordcount_queue.put(None)
            break
            
        # Read file and count words
        try:
            with open(filename, 'r') as f:
                content = f.read()
                
            # Count words in the file
            for word in content.split():
                if word in batch_counts:
                    batch_counts[word] += 1
                else:
                    batch_counts[word] = 1
                    
            files_processed += 1
            
            # If batch size reached, put counts in queue and reset
            if files_processed >= batch_size:
                wordcount_queue.put(batch_counts)
                batch_counts = {}
                files_processed = 0
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    return

def get_top10(counts):
    """
    Determines the 10 words with the most occurrences.
    Ties can be solved arbitrarily.

    Parameters:
    - counts, dictionary: a mapping from words (str) to counts (int)
    
    Return value:
    A list of (count,word) pairs (int,str)
    """
    return sorted([(count, word) for word, count in counts.items()], reverse=True)[:10]

def merge_counts(out_queue, wordcount_queue, num_workers):
    """
    Merges the counts from the queue into a global dictionary. 
    Quits when num_workers Nones have been encountered.

    Parameters:
    - out_queue, multiprocessing queue: queue to output results
    - wordcount_queue, multiprocessing queue: queue that contains word count dictionaries
    - num_workers, int: number of workers (i.e., how many Nones to expect)

    Return value: None
    """
    global_counts = {}
    none_count = 0
    
    while none_count < num_workers:
        counts = wordcount_queue.get()
        
        # Check for end of input signal
        if counts is None:
            none_count += 1
            continue
            
        # Merge counts into global dictionary
        for word, count in counts.items():
            if word in global_counts:
                global_counts[word] += count
            else:
                global_counts[word] = count
    
    # Compute checksum and top10
    checksum = compute_checksum(global_counts)
    top10 = get_top10(global_counts)
    
    # Put results in out_queue
    out_queue.put((checksum, top10))
    
    return

def compute_checksum(counts):
    """
    Computes the checksum for the counts as follows:
    The checksum is the sum of products of the length of the word and its count

    Parameters:
    - counts, dictionary: word to count dictionary

    Return value:
    The checksum (int)
    """
    return sum(len(word) * count for word, count in counts.items())

def run_experiment(path, num_workers, batch_size):
    """
    Run the word counting experiment with the specified number of workers and batch size.
    
    Parameters:
    - path: path to directory with text files
    - num_workers: number of worker processes
    - batch_size: batch size for processing files
    
    Returns:
    - execution time in seconds
    - checksum
    - top10 words
    """
    start_time = time.time()
    
    # Create queues
    filename_queue = mp.Queue()
    wordcount_queue = mp.Queue()
    out_queue = mp.Queue()
    
    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=count_words_in_file, 
                      args=(filename_queue, wordcount_queue, batch_size))
        p.start()
        workers.append(p)
    
    # Start merger process
    merger = mp.Process(target=merge_counts, 
                       args=(out_queue, wordcount_queue, num_workers))
    merger.start()
    
    # Feed filenames into the queue
    for filename in get_filenames(path):
        filename_queue.put(filename)
    
    # Signal end of input
    for _ in range(num_workers):
        filename_queue.put(None)
    
    # Get results from out_queue
    checksum, top10 = out_queue.get()
    
    # Wait for all processes to finish
    for worker in workers:
        worker.join()
    merger.join()
    
    execution_time = time.time() - start_time
    
    return execution_time, checksum, top10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counts words of all the text files in the given directory')
    parser.add_argument('-w', '--num-workers', help='Number of workers', default=1, type=int)
    parser.add_argument('-b', '--batch-size', help='Batch size', default=10, type=int)
    parser.add_argument('path', help='Path that contains text files')
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
    
    # Test with different numbers of workers
    worker_counts = [64, 1,32,2,8,16,4]
    times = []
    checksums = []
    
    print(f"Running experiments with batch size = {batch_size}")
    
    for w in worker_counts:
        if w <= mp.cpu_count() or True:  # Allow testing with virtual cores
            print(f"Testing with {w} workers...")
            execution_time, checksum, top10 = run_experiment(path, w, batch_size)
            times.append(execution_time)
            checksums.append(checksum)
            print(f"  Time: {execution_time:.4f} seconds")
            print(f"  Checksum: {checksum}")
            print(f"  Top 10 words: {top10}")
    
    # Report results with 64 workers
    if 64 in worker_counts:
        idx = worker_counts.index(64)
        print(f"\nTotal absolute running time with 64 workers: {times[idx]:.4f} seconds")
    
    # Calculate speedup
    base_time = times[0]  # Time with 1 worker
    speedups = [base_time / t for t in times]
    
    # Plot speedup
    plt.figure(figsize=(10, 6))
    plt.plot(worker_counts[:len(times)], speedups, marker='o')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title(f'Speedup vs Number of Workers (Batch Size = {batch_size})')
    plt.grid(True)
    
    # Add ideal speedup line for comparison
    ideal_speedups = [w for w in worker_counts[:len(times)]]
    plt.plot(worker_counts[:len(times)], ideal_speedups, 'r--', label='Ideal Speedup')
    plt.legend()
    
    plt.savefig('speedup_plot.png')
    plt.show()
    
    # Verify all checksums are the same
    if len(set(checksums)) > 1:
        print("WARNING: Checksums differ between runs!")
    
    print(f"\nBatch size used: {batch_size}")
    print("Batch size selection rationale: This batch size balances the overhead of queue operations")
    print("with the need to keep the merger process busy. Larger batch sizes reduce communication")
    print("overhead but may delay the merger process, while smaller batch sizes provide more")
    print("frequent updates to the merger but increase communication overhead.")
