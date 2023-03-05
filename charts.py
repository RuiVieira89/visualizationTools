
import xlwings as xw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class waterfall_chart:

    def __init__(self, run=False):

        self.tab = "Data Visualization"
        self.name = "Plots waterfall chart"
        self.comment = "Creates a plot with charts_waterfall_excel_layout_example.xlsx template"

        if run:
            self.plot()

    def plot(self, df=[], title=''):
        print("Running waterfall_chart.plot")
        # Define the data for the chart
        if df != []:
            data = df
        else:
            data = xw.load(index=False)
        
        # Create a list to store the cumulative sum
        cumulative_sum = [0]
        
        # Create a list to store the colors for each bar
        colors = []
        
        # Loop through each row in the data
        for i in range(len(data)):
            if data[data.columns[2]][i] == "abs":
                cumulative_sum.append(data[data.columns[1]][i])
                colors.append("blue")
            elif data[data.columns[2]][i] == "rel":
                cumulative_sum.append(cumulative_sum[i] + data[data.columns[1]][i])
                if data[data.columns[1]][i] > 0:
                    colors.append("green")
                else:
                    colors.append("red")
            elif data[data.columns[2]][i] == "total":
                cumulative_sum.append(data[data.columns[1]][i])
                colors.append("white")
        
        # Plot the chart
        fig, ax = plt.subplots()
        ax.bar(data[data.columns[0]], data[data.columns[1]].values, 
        align='center', color=colors, bottom=cumulative_sum[:-1])
        
        total_data = data[data[data.columns[2]]=='total']

        ax.bar(total_data[total_data.columns[0]],
        total_data[total_data.columns[1]].values,
        color="blue")
        #ax.plot(range(len(data)), cumulative_sum, color='red')

        for i in range(len(data)):
            if data[data.columns[2]][i] == "abs" or data[data.columns[2]][i] == "total":
                ax.text(x=i, y=data[data.columns[1]][i]+0.25,
                s=data[data.columns[1]][i], ha="center")
            elif data[data.columns[2]][i] == "rel":
                if data[data.columns[1]][i] > 0:
                    y_pos = cumulative_sum[i + 1] + 0.25
                    ha = 'center'
                else:
                    y_pos = cumulative_sum[i + 1] - 0.75
                    ha = 'center'
                ax.text(x=i, y=y_pos, s=data[data.columns[1]][i], ha=ha)
    

        # Add labels and title to the chart
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_title(title)

        plt.xticks(rotation=90)

        plt.tight_layout()

        # Show the chart
        plt.show()


class distributions:

    def __init__(self, run=False):
        self.tab = "Data Visualization"
        self.name = "Plots distributions"
        self.comment = "Creates a plot of n numpy arrays"

        if run:
            self.plot_distribution()
            self.grid_plot_dist()

    def plot_distribution(self, data=[], onlyKDE=False):
        print("Running distributions.plot_distribution")
        # data = [np.random.normal(0, 1, 100), 
        # np.random.normal(3, 2, 100), 
        # np.random.normal(3, 2, 100)]
        # data in n numpy arrays

        if len(data) < 1:
            data = xw.load(index=False)
            data = data.values.reshape(1,-1)

        
        colors = sns.color_palette("husl", n_colors=len(data))
        for i in range(len(data)):
            d = data[i]
            color = colors[i]
            # Plot the histogram
            if onlyKDE:
                sns.kdeplot(d, color=color)
            else:
                sns.histplot(d, color=color, kde=True)
            
        plt.show()

    def grid_plot_dist(self, data=[], small_limit=15, hue='dataset'):
        print("Running distributions.grid_plot_dist")

        if len(data) < 1: 
            data = xw.load(index=False)

        try:
            flag = len(data) > small_limit
        except Exception:
            flag = False 
        
        g = sns.PairGrid(data, hue=hue)

        if flag:
            g.map_upper(sns.histplot)
        else:
            g.map_upper(sns.scatterplot)

        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)

# Example usage
# data = [np.random.normal(0, 1, 100), np.random.normal(3, 2, 100), np.random.normal(3, 2, 100)]
# dist = distributions().plot_distribution(data)
