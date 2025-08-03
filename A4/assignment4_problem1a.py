#!/usr/bin/env python3

import time
import argparse
import findspark
findspark.init()
from pyspark import SparkContext

def mapper(line):
    user_follow = line.split(':')
    user = user_follow[0].strip()
    follow = user_follow[1].strip().split()
    return (user, len(follow))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = \
                                    'Compute Twitter follows.')
    parser.add_argument('-w','--num-workers',default=1,type=int,
                            help = 'Number of workers')
    parser.add_argument('filename',type=str,help='Input filename')
    args = parser.parse_args()

    start = time.time()
    sc = SparkContext(master = f'local[{args.num_workers}]')

    lines = sc.textFile(args.filename)
    header = lines.first()
    data = lines.map(mapper).reduceByKey(lambda x, y: x+y)
    no_of_user = data.count()
    total_followed = data.values().sum()
    average = total_followed/no_of_user
    max_follow = data.max(key= lambda x: x[1])[0]
    max_follow_times = data.max(key= lambda x: x[1])[1]
    follow_no_one = data.filter(lambda line: line[1] == 0).count()
    
    end = time.time()
    
    total_time = end - start

    # the first ??? should be the twitter id
    print(f'max follows: {max_follow} follows {max_follow_times}')
    print(f'users follow on average: {average}')
    print(f'number of user who follow no-one: {follow_no_one}')
    print(f'num workers: {args.num_workers}')
    print(f'total time: {total_time}')

