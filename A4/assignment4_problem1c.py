#!/usr/bin/env python3

import time
import argparse
import findspark
findspark.init()
from pyspark import SparkContext


def mapper(line):
    user_follow = line.split(':')
    user = user_follow[0].strip()
    follows = user_follow[1].strip().split()
    return [(user, 0)] + [(follow,1) for follow in follows]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = \
                                    'Compute Twitter followers.')
    parser.add_argument('-w','--num-workers',default=1,type=int,
                            help = 'Number of workers')
    parser.add_argument('filename',type=str,help='Input filename')
    args = parser.parse_args()

    start = time.time()
    sc = SparkContext(master = f'local[{args.num_workers}]')

    lines = sc.textFile(args.filename)

    data = lines.flatMap(mapper).reduceByKey(lambda x,y:x+y)

    total_no_user = data.count()
    total_no_follower = data.values().sum()

    average = total_no_follower / total_no_user

    most_follower_id = data.max(key= lambda x: x[1])[0]
    most_follower_times = data.max(key= lambda x: x[1])[1]

    no_follower = data.filter(lambda x:x[1] == 0).count()

    end = time.time()
    
    total_time = end - start

    # the first ??? should be the twitter id
    print(f'max followers: {most_follower_id} has {most_follower_times} followers')
    print(f'followers on average: {average}')
    print(f'number of user with no followers: {no_follower}')
    print(f'num workers: {args.num_workers}')
    print(f'total time: {total_time}')

