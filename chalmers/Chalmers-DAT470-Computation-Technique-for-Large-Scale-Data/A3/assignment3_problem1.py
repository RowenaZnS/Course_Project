#!/usr/bin/env python3

from mrjob.job import MRJob
import csv
import os
from io import StringIO

class MRMineral(MRJob):
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
        if key in ["Prime Capella","Alpha Cancri","Gamma Sagittae","Beta Lyrae", "Alpha Geminorum"]:
            yield (key, int(total_value))

if __name__ == '__main__':
    MRMineral().run()


