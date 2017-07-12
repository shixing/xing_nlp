# ls *.sh | python merge_jobs.py 2/4

import os
import sys

head4 = """
#!/bin/bash
#PBS -q isi80
#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=16:gpus=4:shared

"""

head2 = """
#!/bin/bash
#PBS -q isi
#PBS -l walltime=100:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared
"""



def main():
    n = int(sys.argv[1])
    jobs = []
    for job in sys.stdin:
        job = job.strip()
        job = os.path.abspath(job)
        jobs.append(job)
        
    
    head = head4
    if n == 2:
        head = head2
    
    for i in xrange((len(jobs)+1)/n):
        f = open("combine{}.sh".format(i),'w')
        f.write(head+"\n")
        for j in xrange(n):
            k = i * n + j
            if k >= len(jobs):
                continue
            f.write("bash {} {} &\n".format(jobs[k],j))
        f.write("wait;\n")
        f.close()

main()
        
