
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class ProcessCapability:
    def __init__(self, data, spec_upper, spec_lower, run=True):
        """
        data: the dataset (numpy array or pandas DataFrame)
        spec_upper: the upper specification limit
        spec_lower: the lower specification limit
        """

        self.data = np.array(data)

        try:
            self.columns = data.columns
        except:
            self.columns = None

        self.spec_upper = spec_upper
        self.spec_lower = spec_lower

        if run:
            self.calculate_indices()
            self.plot_distribution()
            self.print_indices()

    def calculate_indices(self):
        self.mean = np.mean(self.data, axis=0)
        self.std_dev = np.std(self.data, ddof=1, axis=0)
        self.cp = (self.spec_upper - self.spec_lower) / (6 * self.std_dev)
        self.cpk = np.minimum((self.spec_upper - self.mean) / (3 * self.std_dev),
                       (self.mean - self.spec_lower) / (3 * self.std_dev))
        self.ppk = np.minimum((self.spec_upper - self.mean) / (3 * np.sqrt(np.mean((self.data - self.mean) ** 2, axis=0))),
                       (self.mean - self.spec_lower) / (3 * np.sqrt(np.mean((self.data - self.mean) ** 2, axis=0))))

    def plot_distribution(self, show=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, 
                                       #gridspec_kw={'height_ratios': [3, 1]}
                                       )
        _, bins, patches = ax1.hist(self.data, # n, bins, patches
                                    #bins=30, 
                                    label=self.columns,
                                    density=True, 
                                    alpha=0.7)

        ax1.axvline(self.spec_upper, color='r', linestyle='--', linewidth=2)
        ax1.axvline(self.spec_lower, color='r', linestyle='--', linewidth=2)
        _, y_max = ax1.get_ylim()
        ax1.annotate(f'Upper Spec = {self.spec_upper}', xy=(self.spec_upper, 0), 
                     xytext=(self.spec_upper, y_max),
                      #arrowprops=dict(facecolor='black', shrink=0.05), 
                      fontsize=12)
        ax1.annotate(f'Lower Spec = {self.spec_lower}', xy=(self.spec_lower, 0), 
                     xytext=(self.spec_lower, y_max),
                      #arrowprops=dict(facecolor='black', shrink=0.05), 
                      fontsize=12)

        # kde plot
        x = np.linspace(min(bins), max(bins), 100)
        for i in range(self.data.shape[-1]):
            try:
                y = norm.pdf(x, loc=self.mean[i], scale=self.std_dev[i])
            except:
                y = norm.pdf(x, loc=self.mean, scale=self.std_dev)

            try:
                color = patches[i][0].get_facecolor()
            except:
                color = 'black'

            ax1.plot(x, y, color=color, 
                     linestyle='dashed', linewidth=2)

        ax1.set_xlabel('Values', fontsize=15)
        ax1.set_ylabel('Probability density', fontsize=15)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.legend()

        try:
            Table = [["Cp", f"{self.cp:.2f}"],
                    ["Cpk", f"{self.cpk:.2f}"],
                    ["Ppk", f"{self.ppk:.2f}"],
                    ["mean", f"{self.mean:.2f}"],
                    ["std dev", f"{self.std_dev:.2f}"],
                    ["n_samples", f"{len(self.data):.0f}"],
                    ]
        except:
            Table = [["Cp", list(map('{:.2f}'.format,self.cp))],
                    ["Cpk", list(map('{:.2f}'.format,self.cpk))],
                    ["Ppk", list(map('{:.2f}'.format,self.ppk))],
                    ["mean", list(map('{:.2f}'.format,self.mean))],
                    ["std dev", list(map('{:.2f}'.format,self.std_dev))],
                    ["n_samples", f"{len(self.data):.0f}"],
                    ]

        ax2.axis('off')
        if len(self.data.shape) < 2:
            Table = np.transpose(Table)

        table = ax2.table(cellText=Table, 
                  #colWidths=[0.6, 0.6], 
                  cellLoc='center', 
                  loc='upper center',
                  #bbox=[0, -0.3, 1, 0.2],
                  #fontsize=25
                  )
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        plt.subplots_adjust(hspace=0.4)
        
        plt.tight_layout()

        if show:
            plt.show()

    def print_indices(self, print=False):
        if print:
            print(f"Process Capability Index (Cp): {self.cp:.2f}")
            print(f"Process Performance Index (Cpk): {self.cpk:.2f}")
            print(f"Process Performance Index (Ppk): {self.ppk:.2f}")


if __name__ == '__main__':

    #data = [98, 102, 99, 100, 101, 97, 103, 98, 102, 99]

    from utils import generate_random_data

    data = generate_random_data(lower_lim=1, upper_lim=20, size=(15, 5)).abs()

    # Load iris dataset
    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    spec_upper = 5
    spec_lower = 1

    pc = ProcessCapability(df[df.columns[1]], 
                        spec_upper, spec_lower)

    plt.show()

    print('Énd')