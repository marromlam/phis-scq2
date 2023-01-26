#!/usr/bin/env python3

import sys
import htcondor
from os import makedirs
from os.path import join
from uuid import uuid4
import os

from snakemake.utils import read_job_properties


jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)
# print(job_properties)

UUID = uuid4()  # random UUID
# jobDir = '/logs/condor/{}_{}'.format(job_properties['jobid'], UUID)

jobDir = os.path.join(os.environ['HOME'],
                      f"logs/condor/{job_properties['jobid']}_{UUID}")
makedirs(jobDir, exist_ok=True)
shit = os.path.join(jobDir, 'condor.job')
os.system(f"cp {jobscript} {shit}")
sub = htcondor.Submit({
    'executable':   '/bin/bash',
    'arguments':    jobscript,
    'max_retries':  '0',
    'log':          join(jobDir, 'condor.log'),
    'output':       join(jobDir, 'condor.out'),
    'error':        join(jobDir, 'condor.err'),
    'getenv':       'True',
    # TODO:
    'request_cpus': str(min(job_properties['threads'], 64)),
})

request_memory = job_properties['resources'].get('mem_mb', None)
if request_memory is not None:
    sub['request_memory'] = str(2*request_memory)

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    clusterID = sub.queue(txn)

# print jobid for use in Snakemake
print('{}_{}_{} job at {}'.format(job_properties['jobid'], UUID, clusterID, shit))
