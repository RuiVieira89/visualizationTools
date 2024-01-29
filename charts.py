
import xlwings as xw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


class PairGridCorr_plot:
    def __init__(self, data, hue=None):
        self.data = data
        self.hue = hue
        self.g = None
        
    def _get_col_name(self, col):
        # replace underscores with spaces and 
        # capitalize first letter of each word
        return col.replace('_', ' ').title()
        
    def _annotate_corr(self, x, y, hue=None, **kwargs):
        # Group the data by hue category
        if hue is not None:
            grouped = self.data.groupby(hue)
        else:
            grouped = [(None, self.data)]

        # Annotate each subplot for each hue category
        for i, (name, group) in enumerate(grouped):
            #color = palette[i]
            ax = plt.gca()

            # A statistical hypothesis test is a method of 
            # statistical inference used to decide whether 
            # the data at hand sufficiently support a particular 
            # hypothesis. Hypothesis testing allows us to make 
            # probabilistic statements about population parameters.

            # p-value --> probability of obtaining test results 
            # at least as extreme as the result actually observed, 
            # under the assumption that the null hypothesis is 
            # correct. A very small p-value means that such an 
            # extreme observed outcome would be very unlikely 
            # under the null hypothesis.

            # r --> measure of linear correlation between 
            # two sets of data 
            r, p = stats.pearsonr(group[x.name], group[y.name])

            # Annotate the subplot
            if hue is not None:
                ax.annotate(f"{name} r={r:.2f} p={p:.2f}", 
                            xy=(0.1, 0.9 - i/10), 
                            xycoords=ax.transAxes, 
                            **kwargs
                            )
            
            else:
                ax.annotate(f"r={r:.2f} p={p:.2f}", 
                            xy=(.1, .9), 
                            xycoords=ax.transAxes, 
                            **kwargs
                            )
    
    def plot(self):
        g = sns.PairGrid(self.data, hue=self.hue)
        g.map_upper(sns.scatterplot)
        g.map_diag(sns.histplot, kde=True)
        g.map_lower(self._annotate_corr, hue=self.hue)
        g.map_lower(sns.kdeplot)
        g.add_legend()
        self.g = g

        return g


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
# Example usage
# data = [np.random.normal(0, 1, 100), np.random.normal(3, 2, 100), np.random.normal(3, 2, 100)]
# dist = distributions().plot_distribution(data)
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

        
        # Plot the histogram
        if onlyKDE:
            sns.kdeplot(data, kde=True)
        else:
            sns.histplot(data)
            
        plt.show()


def pareto_plot(dataframe, category_column, value_column):
# Example usage:
# Assuming df is your Pandas DataFrame with categorical and numerical columns
# where 'Category' is the categorical column and 'Count' is the numerical column.
# Replace these column names according to your DataFrame structure.

# Example DataFrame creation:
# data = {'Category': ['A', 'B', 'C', 'D', 'E'],
#         'Count': [30, 25, 20, 15, 10]}
# df = pd.DataFrame(data)

# Call the function to plot the Pareto diagram
# pareto_plot(df, 'Category', 'Count')

    # Sort the DataFrame by the specified value_column in descending order
    sorted_df = dataframe.sort_values(by=value_column, ascending=False)
    
    # Calculate cumulative percentage
    sorted_df['cum_percentage'] = (sorted_df[value_column].cumsum() / sorted_df[value_column].sum()) * 100
    
    # Plot the Pareto diagram
    fig, ax1 = plt.subplots()

    # Bar plot for values
    ax1.bar(sorted_df[category_column], sorted_df[value_column], color='b')
    ax1.set_xlabel(category_column)
    ax1.set_ylabel(value_column, color='b')
    ax1.tick_params('y', colors='b')

    # Line plot for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(sorted_df[category_column], sorted_df['cum_percentage'], color='r', marker='o')
    ax2.set_ylabel('Cumulative Percentage', color='r')
    ax2.tick_params('y', colors='r')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Title for the plot
    plt.title('Pareto Diagram')

    # Show the plot
    #plt.show()

'''UTIL'''
def count_categories(dataframe, category_column, label='Count'):
# Example usage:
# Assuming df is your Pandas DataFrame with a categorical column 'Category'.
# Replace 'Category' with the actual column name in your DataFrame.

# Example DataFrame creation:
# data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'C', 'B']}
# df = pd.DataFrame(data)

# # Call the function to count categories
# count_df = count_categories(df, 'Category', label='yup')

# # Print the resulting DataFrame
# print(count_df)
# pareto_plot(count_df, 'Category', 'yup')

    # Count occurrences of each category
    category_counts = dataframe[category_column].value_counts().reset_index()
    category_counts.columns = [category_column, label]

    return category_counts


def pie_chart(dataframe, category_column, value_column):
# Example usage:
# Assuming df is your Pandas DataFrame with categorical data and their frequencies.
# Replace 'Category' and 'Count' with the actual column names in your DataFrame.

# Example DataFrame creation:
# data = {'Category': ['A', 'B', 'C', 'D'],
#         'Count': [30, 25, 20, 15]}
# df = pd.DataFrame(data)

# # Call the function to plot the pie chart
# pie_chart(df, 'Category', 'Count')
# plt.show()

    # Extract unique categories and their corresponding counts
    categories = dataframe[category_column]
    counts = dataframe[value_column]

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=categories, 
            autopct='%1.1f%%', 
            startangle=140, 
            #shadow=True
            )
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

    # Title for the plot
    plt.title('Pie Chart')

    # Show the plot
    #plt.show()



