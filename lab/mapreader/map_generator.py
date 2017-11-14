import deepmind_lab
from multiprocessing import Process, Pipe
import os

def worker(i):
    env = deepmind_lab.Lab(
        'random_maze',
        ['RGB_INTERLACED'],
        config={
        'fps': str(60),
        'width': str(84),
        'height': str(84)
        })

    for episode in range(1):
        for seed in range(i*5,(i+1)*5):
            if not os.path.exists('/home/oliver/dmlab_level_data/map_' + str(-40) + '_' + str(seed) + '.txt'):
                env.reset(-40,seed)

for i in range(4):
    proc = Process(target=worker, args=(i,))
    proc.start()