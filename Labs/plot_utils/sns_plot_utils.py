import matplotlib.pyplot as plt
import seaborn as sns


def plot_func_sns(plot_df, func="e^x"):
    plt.figure(figsize=(12, 6), dpi=100)
    sns.lineplot(data=plot_df, x="x", y="y")
    plt.xlabel("$x$")
    plt.ylabel(f"${func}$")
    plt.title(f"$f(x) = {func}$")
    plt.grid(True)
    plt.show()
