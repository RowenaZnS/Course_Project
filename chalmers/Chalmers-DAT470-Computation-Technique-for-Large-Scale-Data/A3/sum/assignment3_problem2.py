#!/usr/bin/env python3

from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
from io import StringIO


class MRMineral(MRJob):

    def configure_args(self):
        super(MRMineral, self).configure_args()
        self.add_passthru_arg('-k', type=int, default=1)
        csv.reader()
    def mapper(self, _, line):
        if "constellation" in line.lower():
            return
        fields = line.split(",")

        constellation = fields[0]
        star = fields[1]
        mineral_value = float(fields[5])
        star_system = star + " " + constellation
        yield (star_system, int(mineral_value))

    def reducer(self, key, values):
        total_value = sum(values)
        yield None, (total_value, key)

    def reducer_find_top_k(self, _, star_system_values):
        sorted_values = sorted(star_system_values, reverse=True, key=lambda x: x[0])
        for value, star_system in sorted_values[:self.options.k]:
            yield star_system, value

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer),
            MRStep(reducer=self.reducer_find_top_k)
        ]


if __name__ == '__main__':
    MRMineral().run()


