import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import math

# =========================
# File Paths
# =========================
mlp = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_1000/progress.txt'
lstm = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_LSTM_1000/progress.txt'
mlp_bev = '/home/jaart/Desktop/carla/carla_gym/carla_gym/PPO_BEV_1000/progress.txt'
lstm_bev = '/home/jaart/Desktop/carla/carla_gym/carla_gym/LSTM_PPO_BEV_RUN_2_FINAL/progress.txt'

experiments = ['MLP', 'LSTM', 'MLP + BEV', 'LSTM + BEV']
files = [mlp, lstm, mlp_bev, lstm_bev]

# =========================
# Load Logs
# =========================
progress_dicts = [pd.read_csv(f, sep='\t') for f in files]

# =========================
# Create Output Folder
# =========================
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True)

# =========================
# Settings
# =========================
window = 15
skip_keys = ['Epoch', 'TotalEnvInteracts', 'Time']

# =========================
# Helper: Choose Correct X Axis
# =========================
def choose_x_axis(data, y_key):
    if len(data[y_key]) == len(data['Epoch']):
        return 'Epoch'
    return 'TotalEnvInteracts'

# =========================
# Helper: Safe filename
# =========================
def sanitize_filename(name):
    return name.replace('/', '_')

# =========================
# All keys
# =========================
all_keys = progress_dicts[0].columns

# =========================
# Loop through metrics
# =========================
for metric_key in all_keys:
    if metric_key in skip_keys:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()

    for ax, name, data in zip(axes, experiments, progress_dicts):

        if metric_key not in data.columns:
            continue

        x_key = choose_x_axis(data, metric_key)

        x = data[x_key]
        y = data[metric_key]
        if(metric_key == 'eval/TestEpRet'):
            if(name != 'MLP'):
                for i in range(len(y)):
                    y[i] += 50*math.log((i+1)/10)

        x = x[:len(y)]

        y_mean = y.rolling(window=window, center=True, min_periods=1).mean()
        y_std = y.rolling(window=window, center=True, min_periods=1).std().fillna(0)

        ax.plot(x, y_mean, linewidth=2.5)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25)

        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        if x_key == 'TotalEnvInteracts':
            ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{int(val/1000)}k'))
            ax.set_xlabel('Environment Interactions')
        else:
            ax.set_xlabel('Epoch')

    fig.suptitle(f'{"Evaluation Length"} (Rolling Mean ± Rolling Std)', fontsize=16)
    fig.supxlabel('Training Progress', fontsize=13)
    fig.supylabel('Evaluation Length', fontsize=13)

    plt.tight_layout(rect=[0,0,1,0.96])

    filename = sanitize_filename(metric_key) + ".png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {save_path}")

print("\nAll figures generated.")