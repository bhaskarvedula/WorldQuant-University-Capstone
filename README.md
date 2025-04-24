# WorldQuant-University-Capstone
This is a capstone project for MScFE from WorldQuant University (Group 9184)
## **Concepts**

Reinforcement learning (RL) is a part of machine learning applied to a sequential decision-making process. It involves an agent that seeks to achieve its objective, also referred to as reward, by gathering information from its environment. The interaction of an agent with its environment is accomplished through state, which is the representation of the environment for the agent. Through its interaction with the environment as represented by the state presented to the agent, the agent learns a policy that maximizes its cumulative reward. Policy is an action that an agent takes when presented with a certain state. The key idea is that in a sequence of steps the agent receives a state and takes an action and receives a reward and a new state representation. The agent learns its policy that will maximize the cumulative reward over muti-steps. The core elements of an RL algorithm are the following:

1.   **Agent**. The entity that interacts with the environment and acts based on the state it receives from the environment.

2.   **Environment**. The universe in which the agent exists. Environment provides its current situation to the agent through state, provides rewards based on the action the agent takes and indicates the change in situation through a new state.

3.  **State**: The representation of the environment as provided to the agent.

4. **Action**. This is what an agent takes when provided with a state.

5. **Reward**. This is the feedback that the environment provides to a agent based on the action an agent takes.

6. **Policy**. Policy refers to the action that an agent takes based on the state that the environment offers to the agent.

<BR>
Markov Decision Process (MDP) is the mathematical foundation for RL algorithms and provides an executable framework for RL problems. MDPs are based on Markovian property that the future state is dependent on current state and the action taken by the agent. This assumption greatly simplifies the solving of the RL problems.
<BR>
In this capstone, we use policy-based algorithms that are well suited to the problem we are addressing. PPO, A2C, DDPG, TD3 and SAC are the policy-based algorithms that we would use in this study. We briefly describe these algorithms below:

1.   **Proximal Policy Optimization (PPO).** PPO is an on-policy algorithm that uses data resulting from current policy. Its  key aspect is that it facilitates stable learning by not allowing large updates to policy. It is known for its simplicity, stability and efficiency.

2.  **Advantage Actor-Critic (A2C).** A2C leverages the advantages of both policy-based and value-based algorithms. It has two parts – an actor and a critic. An actor takes the action based on the state presented to it. A critic indicates the value of a state or state-action pair that helps the actor take the appropriate action. However, as an on-policy method, A2C suffers from the curse of sample inefficiency.

3. **Deep Deterministic Policy Gradient (DDPG).** DDPG operates for processes that have continuous actions. It is model-free, and unlike above, it is an off-policy algorithm that uses the actor-critic logic described above.  Also, unlike the above algorithms that have stochastic policies, DDPG has a deterministic policy, which means that for a particular state, it has a particular action. Also, as it is based on neural networks to define policy and value functions, it is sensitive to choice of hyperparameters.

4. **Twin Delayed Deep Deterministic Policy Gradient (TD3).** TD3 is an advancement over DDPG addressing its value over-estimation and instability. It overcomes over-estimation limitations in DDPG by employing two critics and using the lower of them as q-value. The instability issue is overcome by delaying policy updates vis-à-vis ‘critic’ updates and introducing noise in the actions in ‘critic’ updates. Like DDPG, it uses target networks and experience replay for stability.

5. **Soft Actor-Critic (SAC)**. SAC is like actor-critic algorithms operating in continuous actions space but additionally incorporates entropy within the reward structure. This brings about greater stability by encouraging greater exploration. Unlike DDPG and TD3, SAC employs stochastic policy as opposed to deterministic policies. The additional feature, however, introduces greater computing complexity.


## **Data, Transformation and Design**

### *Description of Dataset*
We select a time-period from January 1, 2021, to February 28, 2025, for our study that includes both training and testing (out of sample) periods. This choice of time-period is reflective of medium-term trading agent that needs to be trained on a certain period of time and tested over a relatively medium timeframe (~1 year). Based on the perusal of published literature, about four years of training data should allow our agents to decipher sustained and stable elements in data to trade for the following year without recalibration.
We used the Yahoo Finance website as our main data source as this platform provides the required data in easily extractable form (for programming in Python). Through this website, we gather datasets for each of the chosen stocks that contain daily open price, high price, low price and close price and volumes for all these stocks. We also gather daily index values for NIFTY 50 for the chosen time horizon.
<BR>
<BR>
The portfolio constituents for this analysis are 26 companies that form part of the NIFTY 50 index. Our objective is for the RL agents to output weights of these companies within the portfolio such that at the end of the investment horizon, the value of the portfolio will be  maximized. The rebalancing will be performed daily, and trading of these stocks will be towards ensuring the weights forecast by RL agents. To have a realistic market scenario, we consider a transaction of 0.1% of the transacted amount.
### *Description of Algorithms*
The objective of our study is to gauge the performance of traditional portfolio management techniques before drawing comparisons between them and the RL algorithms. These traditional portfolio management techniques constitute our baseline or benchmark algorithms. The algorithms chosen for this study include Markowitz means-variance theory, De Prado denoising, equal-weighting and Kelly’s Criterion. Based on the outcome of the analysis, we choose the best performing RL and traditional algorithm to be compared against NIFTY buy-and-hold strategy.
<BR>
<BR>
The RL algorithms used for this study include A2C, PPO, TD3, SAC and DDPG and were developed utilizing the FinRL library created by Liu et al. [22] and the Stable-Baselines3 package [23]. For these RL agents, we have defined three different state representations. First are the technical indicators that are typically used by technicians to predict stock price movements. These include Average True Range, Bollinger Bands, On-balance volume, Moving Average Convergence Divergence, Average Directional Movement Index, Simple Moving Average, Exponential Moving Average, Commodity Channel Index, and  Relative Strength Index. We also use lagged returns as the state. For this purpose, we use the lagged returns for 1 day, 2 days, 5 days, 21 days and 30 days. Finally, we train an LSTM model that predicts the return for the following day. Its output will represent ‘states’ for our RL agents.
<BR>
<BR>
We also use Hidden Mark Model to identify and understand regime changes using NIFTY 50 index as the proxy for our portfolio. We also use correlation analysis to understand dependencies and relationships within the stocks in our portfolio.
Data transformations
<BR>
<BR>
Based on the raw data that has been extracted, certain transformations are carried out for the purposes of calibrating and testing these algorithms. These transformations include identifying and addressing missing data, creating lagged returns, covariances, and creating technical indicators.  We also define, train and run an LSTM model and use its output as states of our RL agents. After all transformations have been performed on the dataset, this dataset is split into training and testing (out of sample) sets in the proportion of 75% and 25% to train and evaluate our algorithms. As the our dataset is a financial timeseries, we have split the data into a training time horizon and testing time horizon.
<BR>
<BR>
### *RL setup*
The RL environment is defined as the stock environment that take an action and provides back reward in the nature of portfolio value based on the weights provided (which is action). We use multiple state representations that include technical indicators, lagged returns and LSTM model output. Once an action is taken by the agent, the environment provides the agent with a new state. The reward function is defined as the cumulate gain on the portfolio at the end of investment horizon. We have chosen this reward as we expect it to encapsulate all aspects that ultimately result in maximum portfolio gain.
### *Performance measures*
We use the following measure to gauge the performance of traditional algorithms and RL agents. These measures are commonly used in financial analysis.
*   Cumulative returns
*   Maximum drawdown
*   Sharpe’s ratio
<BR>
<BR>

For analysis of best performing algorithm and RL agent with the buy and hold strategy, we expand the above measures further and include metrics like Sortino’s ratio.


## **Code explanation**

The code written for this capstone has been compartmentalized as multiple section with each section capturing a particular functionality or activity. This is to enable ease of understanding the entire code.


## *Section 1: Python library installations and imports*

### Installations

For this project, we need several Python libraries that are not a part of Google Colab. These include FinRL, stable-baselines and gym libraries for developing reinforcement learning (RL) agents and environment. yfinance the Yahoo Finance source for our data. ta for allowing us to create technical indicators as our 'states' for RL agent. pyfolio for enabling graphical dispaly of performance and comparison metrics. hmmlearn for regime change detection within our data using Hidden Markov Model. tensorflow for training our LSTM models.

### Imports

Imports follow the libraries that we have installed above. stable baseline provides us with various RL agents based on a particular algorithm (DDPG, PPO, TD3, SAC and A2C). Other imports are primarily the usual imports required for a financial project like pandas, numpy, etc.

## *Section 2: Defining functions and classes*
In this section, we define functions and classes that are necessary for this capstone project.

### Functions

In this section, function is defined to create technical indicators as our states. This function takes close prices, open price, high and low price as inputs and outputs the technical indicators of the chosen stock tickers. The function utilizes ta library to achieve this functionality. Most of the remaining functions are defined primarily to achieve denoising technique as part of our benchmark techniques. These functions are based on the previous courses completed at WorldQuant University. Finally, we define few functions to facilitate reporting and analyzing performance statistics.

### Classes
In this section, classes have been defined for the purpose of embedding functionality to our reinforcement learning (RL) agent and environment.

## *Section 3: Defining 'States' for our Reinforcement Learning Agent*

In this section, we define states for our reinforcement learning (RL) agent. States are what the RL environment provides to the RL agent in order to make a decision or take an action. For this capstone, we explore three different state repesentations to explore the effectiveness of state definition on RL agent's performance. We define the three variants of the states that we use for running for RL algorithms

## *Section 4: Defining model parameters*

We now define our model parameters for the various RL agents that we propose to use.

## *Section 5: Defining, extracting, understanding our data and creating training and testing data*

In this section, we define our data and understand the data including correlations, regime changes etc. In this section, we define start date, end date, and stock tickers. we also create the data for training and testing various states we defined earlier.
<br><BR>
Next, we extract and transform our data. We use Yahoo Finance to source our data. Notice that we also introduce 'CASH' as one of the tickers. We do so to allow our agents to move investments from stock to cash and vice versa when there is a bearish or bullish phase when all stocks are expected to behave a symmetric and coordinated manner.
<br><br>
We create data for our state representing technical indicators. We now create a lag returns data for evaluating states that represent lagged returns. Defining state and end dates for our training and testing data.
<br><br>
We now create data for states representing predictions from our LSTM model. For we first create training data to train a LSTM model and then use this model to spew out our 'state' representation. Checking performance of our LSTM output to validate its role as a state for our RL agents. Creating data for LSTM states.Creating traing and testing data for all 'states' of RL agents
<br><br>
Reviewing correlations of returns from our closing price data of the chosen tickers. Analysing regimes in the data (using NIFTY 50 as the proxy)

## *Section 6: Training and testing of traditional and RL agents*

In this section, we train and test all our approaches/agents with the training and testing data that we had created earlier. This section occupies largest code base as not only all traditional and RL agenst are trained and tested, but the performance of RL agents sharing the same states are analysed as part of this section.

## *Section 7: Performance Analysis traditional and RL agents*

We analyse the performance of all trading agents - traditional and RL based in this section. We first evaluate against the training data both in the sample and out of the sample.

## *Section 8: Comparing the buy-and-hold strategy with best performing RL and Traditional algorithms*

In this section, we perform a deep dive analysis of the best performing RL algorithm and traditional algorithms and understand the causes of this outperformance.
