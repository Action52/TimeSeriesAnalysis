import math
import time
import pandas as pd
import numpy as np
import numpy
import random
import decimal
import scipy.linalg
import numpy.random as nrand
import matplotlib.pyplot as plt
import sys
from function import *
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
import altair as alt

custom_colors = ["#948B80", "#625D56", "#312E2B", "#875C62", "#5F4045", "#B79DA1",
                "#417156", "#2E4F3C", "#6D7C73"]


class ModelParameters:
    def __init__(self,
                 all_s0, all_time, all_delta, all_sigma, gbm_mu,
                 jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0, all_r0 = 0):
        # This is the starting asset value
        self.all_s0 = all_s0
        # How long you want to stimulate
        self.all_time = all_time
        # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        self.all_delta = all_delta
        # This is the volatility of the stochastic processes
        self.all_sigma = all_sigma
        # This is the annual drift factor for geometric brownian motion
        self.gbm_mu = gbm_mu
        # This is the probability of a jump happening at each point in time
        self.lamda = jumps_lamda
        # This is the volatility of the jump size
        self.jumps_sigma = jumps_sigma
        # This is the average jump size
        self.jumps_mu = jumps_mu
        # This is the starting interest rate value
        self.all_r0 = all_r0


def get_jumps(jumps,train_len):

    """
    Calculate the jumps in the train data and return the probability for a jump
    """

    jumps.sort()
    jumps_Reg = [round(np.exp(i)-1, 4) for i in jumps]
    print("5 largest positive single day moves ", list(reversed(jumps_Reg[-5:])))
    print("5 largest negative single day moves", jumps_Reg[:5])

    # Probability of jump is simple num_of_jump/total_observation 
    prob_of_jump = len(jumps)/train_len
    print("The probability of jump is", round(prob_of_jump, 3), "%")

    return prob_of_jump

def merton_jump_diffusion(train_df, target, time_to_maturity = 365, paths = 10000):

    """
    Create a jump diffusion model and generate the given number of pathes (default 10000)
    """
    
    train_df['differance'] = np.log(train_df[target]/ train_df[target].shift(-1))
    SD_stock = np.std(train_df['differance'])
    Mean_stock = np.mean(train_df['differance'])
    daily_returns = (train_df['differance'][1:]).tolist()
    jumps = [returns for returns in daily_returns if np.abs(returns - Mean_stock) > 1.96 *SD_stock]
    
    prob_of_jump = get_jumps(jumps,len(train_df))
    
    S0 = train_df.iloc[-1][target]

    # average price movement
    drift = np.mean(train_df['differance'])*np.sqrt(time_to_maturity)

    # non jump price movement SD
    non_jumps = [returns for returns in daily_returns if np.abs(returns - Mean_stock) < 1.96 *SD_stock]
    gbm_sd = np.std(non_jumps)*np.sqrt(time_to_maturity)

    # risk_free rate, using US T-bill since BTC-USD is compared
    rf = 0.0384

    # mean of the jumps
    mean_jump_size = np.mean(jumps)

    # SD of Jumps
    SD_of_jump = np.std(jumps)

    mp = ModelParameters(
            # This is the starting asset value
            all_s0 = S0, 
        
            # How long you want to stimulate
            all_time = time_to_maturity,
        
            # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
            all_delta = 1/time_to_maturity,
        
            # This is the volatility of the stochastic processes
            all_sigma = gbm_sd,
        
            # This is the annual drift factor for geometric brownian motion
            gbm_mu = drift,
        
            # This is the probability of a jump happening at each point in time
            jumps_lamda = prob_of_jump,
        
            # This is the volatility of the jump size
            jumps_sigma = SD_of_jump,
        
            # This is the average jump size
            jumps_mu = mean_jump_size,
        
            # This is the starting interest rate value
            all_r0 = rf)

    jump_diffusion_examples = []
    for i in range(paths):
        jump_diffusion_examples.append(geometric_brownian_motion_jump_diffusion_levels(mp))
    plot_stochastic_processes(jump_diffusion_examples, "Jump Diffusion Geometric Brownian Motion (Merton)")
    
    #if plotting should be done with altair uncomment this and return chart 
    #chart = alt_plotting(jump_diffusion_examples,train_df.index[-1],time_to_maturity)

    return mp, jump_diffusion_examples #,chart


def generate_date_list(start_date, num_days):

    """
    Create a list of dates for a given start date and length
    """

    date_list = []
    current_date = start_date + pd.DateOffset(days=1)

    for _ in range(num_days):
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list



def alt_plotting(jump_diffusion_examples,start_date,time_to_maturity):

    """
    Plot the pathes with altair (for the presentation)
    """

    date_list = generate_date_list(start_date, time_to_maturity)
    paths_df = pd.DataFrame(jump_diffusion_examples).T.reset_index()
    paths_df['date'] = date_list


    cols = paths_df.columns.tolist()  # Get column names as a list
    cols = ['date'] + [col for col in cols if col != 'date']  # Move 'date' to the front
    paths_df = paths_df[cols]  # Reorder columns

    paths_df.drop(columns=['index'], inplace=True)
    paths_df = paths_df.melt(id_vars='date', var_name='path', value_name='stock_price')

    # Renaming columns
    paths_df.columns = ['Date', 'Path', 'Bitcoin Value']

    # Creating Altair chart
    chart = alt.Chart(paths_df).mark_line().encode(
        #x='Time Step',
        x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m')),
        y='Bitcoin Value',
        color='Path:N'
    ).properties(
        title='Jump Diffusion Geometric Brownian Motion (Merton) - 1 year',
        width = 800,
        height = 300
    ).configure_point(
        size = 10
    ).configure_legend(
        title=None,  # Setting title to None to hide legend title
        labelFontSize=0,  # Hiding the legend labels
        symbolSize=0,  # Hiding the legend symbols
        symbolFillColor='white'  # Making the legend background color white to hide it
    )

    #chart.save('graphs/btcusd/arimax_returns.json')

    return chart


def plot_avg_train(jump_diffusion_examples, test_df, target,time_to_maturity = 365):
    
    """
    Plot the test data and the average path of the jump diffusion model
    """

    paths_array = np.array(jump_diffusion_examples)

    # Calculate the median path
    mean_path = np.mean(paths_array, axis=0)

    test_values = test_df[target].iloc[:time_to_maturity].values

    # Plotting the median path
    plt.figure(figsize=(8, 6))
    plt.plot(mean_path, label='Mean Path', color='green')
    # Plotting the first 365 values of the test dataframe
    plt.plot(test_values, label='Test Data', color='blue')

    plt.title('Mean Path of 10,000 Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    r2 = r2_score(test_values, mean_path[:time_to_maturity])
    print(f'R-squared score: {r2}')

    return mean_path


def plot_avg_train_alt(jump_diffusion_examples, test_df, target,time_to_maturity = 365):

    """
    Plot the average path with altair (for the presentation)
    """

    # Convert it into a NumPy array for easier calculations
    paths_array = np.array(jump_diffusion_examples)

    # Calculate the median path
    mean_path = np.mean(paths_array, axis=0)

    test_values = test_df[target].iloc[:time_to_maturity].values

    dates = test_df.index[:time_to_maturity]

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Date': dates,  # Assuming the index contains the dates
        'Mean': mean_path,
        'Actual': test_values
    })

    # Melt the DataFrame for Altair's plotting
    melted_df = plot_df.melt('Date', var_name='Type', value_name='Bitcoin Value')

    # Create the Altair chart
    alt_chart = alt.Chart(melted_df).mark_line().encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('Bitcoin Value:Q', scale=alt.Scale(domain=[16000, 40000])),
        #color='Type:N'
        color=alt.Color('Type:N', scale=alt.Scale(range=custom_colors))
    ).properties(
        width=800,
        height=300,
        title='Mean path of the Mertons Jump Diffusion Model'
    ).configure_point(
        size=10
    )

    # Save the chart
    #alt_chart.save('graphs/btcusd/arimax_returns.json')
    return alt_chart



def get_finalPrices(jump_diffusion_examples):
    """
    Get a list of the final prices of all pathes
    """
    final_prices = []
    for _, price in enumerate(jump_diffusion_examples):
        final_prices.append(price[-1])

    return final_prices



def get_priceDistribution(S0, jump_diffusion_examples):

        """
        Return distribution of the final prices over the generated pathes 
        """
        # Getting the price distrbution
        final_prices = get_finalPrices(jump_diffusion_examples)

        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(final_prices, bins=450, range=[0, S0*4], color='gray')
        for i, patch in enumerate(patches):
            if bins[i] <= S0:
                patch.set_fc('#DA543C')
            else:
                patch.set_fc('#62C180')
        plt.title("Final Price Distribution")
        plt.show()

    
def get_profit_loss(S0, jump_diffusion_examples):

    """
    Return Profit/loss distribution. Comparison of start price and
    final price. 
    """

    # Calculate profit/loss for each path based on buying at S0 and selling at final price
    final_prices = get_finalPrices(jump_diffusion_examples)
    profit_loss_from_S0 = np.array([final_price - S0 for final_price in final_prices])

    # Plotting the profit/loss distribution
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(profit_loss_from_S0, bins=450, range=[-S0*3, S0*3], color='gray')
    for i, patch in enumerate(patches):
        if bins[i] <= 0:
            patch.set_fc('#DA543C')  # Set color for losses
        else:
            patch.set_fc('#62C180')  # Set color for profits
    plt.title("Profit/Loss Distribution from Buying at S0")
    plt.xlabel("Profit/Loss")
    plt.ylabel("Frequency")
    plt.show()

    # Calculate statistics
    mean = np.mean(profit_loss_from_S0)
    minimum = np.min(profit_loss_from_S0)
    maximum = np.max(profit_loss_from_S0)
    q1, median, q3 = np.percentile(profit_loss_from_S0, [25, 50, 75])

    # Print statistics
    print("Mean:", mean)
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("25th percentile (Q1):", q1)
    print("Median (50th percentile):", median)
    print("75th percentile (Q3):", q3)