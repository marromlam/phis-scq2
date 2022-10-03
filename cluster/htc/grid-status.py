#!/usr/bin/env python

import sys
import htcondor
from htcondor import JobEventType
# from os.path import join
import os

def print_and_exit(s):
    print(s)
    exit()


jobID, UUID, clusterID = sys.argv[1].split('_')

# jobDir = '/home3/marcos.romero/logs/condor/{}_{}'.format(jobID, UUID)
jobDir = os.path.join(os.environ['HOME'], f'logs/condor/{jobID}_{UUID}')
jobLog = os.path.join(jobDir, 'condor.log')

failed_states = [
    JobEventType.JOB_HELD,
    JobEventType.JOB_ABORTED,
    JobEventType.EXECUTABLE_ERROR,
]

try:
    jel = htcondor.JobEventLog(os.path.join(jobLog))
    for event in jel.events(stop_after=5):
        if event.type in failed_states:
            print_and_exit('failed')
        if event.type is JobEventType.JOB_TERMINATED:
            if event['ReturnValue'] == 0:
                print_and_exit('success')
            print_and_exit('failed')
except OSError as e:
    print_and_exit('failed')

print_and_exit('running')
