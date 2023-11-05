# Architecture of Code

## Loading in Libraries and Data

To begin, we are going to have to import a number of libraries for our machine learning techniques, including many of the sklearn features. After the libraries have been loaded, the data is also going to be loaded in using `pd.read_csv(filename)` which will return a DataFrame with all of the statistics. 

Now that it is loaded in we can take a look at some of the attributes the data provides including: `year` `player_age` `WAR` and ` salary `. With these stats in mind, we can begin to analyze the data

## Adjusting Salaries for Inflation

Since this model is going to take into account financial compensation of players, we need to make sure that in the future we are properly assigning value to production. Thus, if we are going to predict contracts in today's $ value, we must account for inflation when looking at previous contract values.

The dict `INFLATION_CONVERSION` holds the conversion factors from 2015-2022, so that we can take the `salary` of each player and apply the inflation rate to today from that `year`. Initial reports show that if we average all the salaries (among qualified batters from 2015-2022), we get a value of around $9.5 Million.

This will prove fruitful later on, as we take this average salary, and apply it to how it tracks with `WAR` `age` and other statistics.

## Looking at Trends among Stats

Using `seaborn` and the `pairplot` function, I was able to generate an image that displays all of the pairs of stats up against one another. This is quite a large image, so there is a lot going on here!

![seabornAnalysis](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/d8d4a531-d0b0-4148-8bb4-c14a994cfeab)

There are a few obvious correlations, like `batting_avg` being strongly correlated in the positive direction with `hits`. I am most interesting the `WAR` statistic, paired up with `player_age`. `WAR` is a universal metric that standardizes and objectifies how replaceable the player is. By pairing this up with `player_age` we can track trends in how players get older, when do they start to decline.

![playerAge and WAR](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/8f6fa833-3a57-4de0-b6ac-8fe3813f3795)

## Application of Regression Techniques to the Data

With this trend, I want to create a model that recognizes the trends in this data. Bare boned linear regression seems as if it wouldn't be useful for what I am looking to do. The data suggests a non-linear relationship, as there seems to be an increase in `WAR` with `player_age` until near 30, when it then begins to decline.

So here we go, looking at Chapter 19 in the R&N textbook, 19.7.4 talks about "piecewise-linear nonparametric regression". In particular, what catches my eye is locally-weighted regression. After perusing `sklearn`, I couldn't find anything on this, but I stumbled upon `PolynomialFeatures`, which seemed applicable to this data set. The relationship suggests some sort of uptick until a certain age, until then it goes down, so a 2nd degree polynomial is something that could make sense. 

So I split up my data 80% training, 20% testing like so: 
`X = stats['player_age'] \n`,
`X = np.array(X).reshape(-1,1)`, and
`Y = stats['WAR']`

and then split it: `X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)`

after fitting the model to the training data and then testing it, I was able to produce the following.

![MODELED_playerAge_WAR](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/705ae5bf-2dd0-4676-a896-a996d08a318d)

This is a great start, as it captures the tail off in the end of player's careers. However, the testing doesn't quite capture the initial uptick in player performance at the start of the careers. Thus, I believe it would be wise to look for another technique to try to quantify this trend, or modify the current one in some way.
