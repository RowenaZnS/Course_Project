#!/usr/bin/env python3
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRJobTwitterFollowers(MRJob):
    # The final (key,value) pairs returned by the class should be
    #
    # yield ('most followers id', ???)
    # yield ('most followers', ???)
    # yield ('average followers', ???)
    # yield ('count no followers', ???)
    #
    # You will, of course, need to replace ??? with a suitable expression
    def mapper(self, _, line):
        user_follow = line.split(':')
        user = user_follow[0].strip()
        follows = user_follow[1].strip().split()
        yield (user, 0)
        for follow in follows:
            yield (follow, 1)
    
    # Add combiner to perform local aggregation
    def combiner(self, key, values):
        # Sum up the follower counts locally before sending to reducer
        yield (key, sum(values))
    
    def reducer(self, key, values):
        yield (None, (key, sum(values)))
    
    def reducer2(self, _, values):
        user_list = list(values)
        max_pair = max(user_list, key=lambda x: x[1], default=(None, -1))
        max_id, max_count = max_pair
        no_follow = sum(1 for v in user_list if v[1] == 0)
        total_follow = sum(v[1] for v in user_list)
        total_users = len(user_list)
        yield ('most followers id', max_id)
        yield ('most followers', max_count)
        yield ('average followers', total_follow / total_users if total_users else 0)
        yield ('count no followers', no_follow)
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   combiner=self.combiner, 
                   reducer=self.reducer),
            MRStep(reducer=self.reducer2)
        ]

if __name__ == '__main__':
    MRJobTwitterFollowers.run()
