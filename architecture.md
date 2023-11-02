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
