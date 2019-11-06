import os

def runningInCI():
    return 'CI' in os.environ or 'TRAVIS' in os.environ

