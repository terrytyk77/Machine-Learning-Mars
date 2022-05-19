class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


def plot_correlation_matrix(plt, title, df):
    figure = plt.figure(figsize=(9, 9))
    plt.matshow(df.corr(), fignum=figure.number)
    plt.xticks(range(
        df.select_dtypes(['number']).shape[1]),
        df.select_dtypes(['number']).columns,
        fontsize=14,
        rotation=45)
    plt.yticks(range(
        df.select_dtypes(['number']).shape[1]),
        df.select_dtypes(['number']).columns,
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    plt.show()
