import datetime
import os
import subprocess

h = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).strip()
print("{} Git hash: {}".format(datetime.datetime.now(), h))

with open('config.INI', 'r') as fin:
    print('{} Configuration:\n'.format(datetime.datetime.now()))
    print(fin.read())
    print()

print('{} Preparing data.'.format(datetime.datetime.now()))
os.system('python prepare_data.py')

print('{} Training model.'.format(datetime.datetime.now()))
os.system('python train_model.py')

print('{} Analysing data.'.format(datetime.datetime.now()))
os.system('python analyse_model.py')

print('{} Writing summary.'.format(datetime.datetime.now()))
os.system('python write_json_summary.py')