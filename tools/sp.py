#!/usr/bin/env python

__license__ = \
    """Copyright 2019 West University of Timisoara
    
       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at
    
           http://www.apache.org/licenses/LICENSE-2.0
    
       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

import argparse
import multiprocessing
import subprocess
import sys
import threading
from queue import Queue

num_procs = multiprocessing.cpu_count()
parser = argparse.ArgumentParser(description='Process some data')
parser.add_argument('--num-workers', type=int, help='Number of processes (%d)' % num_procs, default=num_procs)
parser.add_argument("--handler", type=str, help='Program to be called on each line')
parser.add_argument("--output", type=argparse.FileType('w'), default=sys.stdout, help="Destination file")
args = parser.parse_args()


class Worker(threading.Thread):
    def __init__(self, cmd, idx, in_q, out_q):
        self.idx = idx
        self._in_q = in_q
        self._out_q = out_q
        self._cmd = cmd
        threading.Thread.__init__(self)

    def run(self):
        # print ("Worker %d starting..." % self.idx)
        while True:
            work = self._in_q.get()
            if work is None:
                self._in_q.put(None)
                break

            line_num, line = work
            env = {
                '_SAGE_SP_WORKER': str(self.idx),
                '_SAGE_SP_LINE_NUM': str(line_num)
            }

            ret = subprocess.run(self._cmd, shell=True, input=bytes(line, "utf8"), stdout=subprocess.PIPE, env=env)
            if ret.returncode == 0:
                status = "SUCCESS"
            else:
                status = "ERROR"
            result = "%s:%d-%d:%s" % (status, self.idx, line_num, ret.stdout.decode("utf8"))
            self._out_q.put(result)

        # print ("Worker %d finishing..." % self.idx)
        self._out_q.put(None)


if __name__ == "__main__":
    input_q = Queue()
    output_q = Queue()
    for i in range(0, args.num_workers):
        Worker(args.handler, i, input_q, output_q).start()

    q = Queue()
    line_num = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        line_num += 1
        if not line:
            continue
        msg = (line_num, line)
        input_q.put(msg)
    input_q.put(None)  # Stop

    finished_workers = 0
    while True:
        result = output_q.get()
        if result is None:
            finished_workers += 1
        if finished_workers == args.num_workers:
            break
        if result is not None:
            args.output.write(result)
