from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import matplotlib.pyplot as plt

mlp = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_1000/progress.txt'
lstm = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_LSTM_1000/progress.txt'
mlp_bev = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_BEV_1000/progress.txt'
lstm_bev = '/home/jaart/Desktop/carla/carla_gym/carla_gym/LSTM_BEV_1000/progress.txt'

experiments = ['mlp', 'lstm', 'mlp bev', 'lstm bev']
files = [mlp, lstm, mlp_bev, lstm_bev]

for i in range(len(files)):

    progress_dict = {}


    with(open(files[i], 'r')) as f:
        lines = [line.strip().split('\t') for line in f.readlines()]


    headers = lines[0]
    for h in headers:
        progress_dict[h] = []

    for row in lines[1:]:
        padded_row = row + [None] * (len(headers) - len(row)) #Pad with Nones
        for h, v in zip(headers, padded_row):
            progress_dict[h].append(float(v) if v is not None else v)
    files[i] = progress_dict


progress_dict = files[0]
print(len(progress_dict['eval/TestEpRet']))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    progress_dict['TotalEnvInteracts'],
    progress_dict['learner/Entropy'],
    linewidth=2,
)

# Fewer ticks on both axes
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Pretty formatting for large x values (e.g. 200k instead of 200000)
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f'{int(x/1000)}k')
)

ax.set_xlabel('Total Environment Interactions')
ax.set_ylabel('Evaluation Episode Return')
ax.set_title('Training Progress')

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

listA = [0, 1, 2, 3]
listB = listA[0:10]
print(listB)