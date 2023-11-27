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

## Further Exploration of Techniques

Not deterred from the mediocre results to this point, I kept digging deeper into Scikit-Learn, trying to find a good way to work with my data. After some time, I stumbled upon the `Pipeline` tool which helps streamline the data transformation process to allow for polynomial regression. Playing around with it, I was able to obtain graphs that looked the part finally.

![Graph](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/d78b279d-b412-4c54-b38e-5b8ba071e70c)

Still, a couple things to note, because of the test train split, our testing set will omit players at the ends of the age gap. There just simply aren't enough points for 38+ year olds and 20- year olds to guarantee they'll be in the testing set. So, as I move forward with the contract predictor, it is key to remember that this regression won't cover really old players or really young players accurately.

## Contract Evaluation

Now that there is a tangible model with sensible results, it is feasible to start the contract evaluation. First, I want to calculate the average salary among qualified batters as a baseline of what players are being paid. At the same time I want to calculate the average WAR for a player that is a qualified batter. These come out to around `$9.5 M` for the average salary (adjusted for inflation) and `3.05` average WAR among qualified batters from 2015-2022. This paints the picture that teams are paying around `$3.1 M` per Win Above Replacement.

![image](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/5847e1fe-2dd0-488f-aa82-90d71173c006)

With this readily available, my vision is to take players' WAR from last season, and apply the relationship found by the regression model to predict their WAR in future seasons. With their predicted WAR, I can then take the $/WAR ratio, and apply it to the player's future production. Thus, the contract would be made.

First, I made a cell block that get's the `playerFirst` and `playerLast` names from the user. With this I was able to obtain `PWAR` which is the player's WAR from the 2022 season, along with their `AGE`.

![image](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/0d95292d-a10e-461a-b25e-db4b77950bcc)

Next, I made a cell block such that I enter how many years into the future I am going to look. I then take the regression relationships found earlier, and apply them to each of the future seasons. So from that `AGE` season, I can see the percent changes.

![image](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/3841972c-d51b-48d9-b277-f609b33fb2c8)

The percentages will now be applied to the `PWAR` found earlier, and store all of the predicted WARS in the `playerPrediction` list. 

![image](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/5a003fd0-c54f-4be2-add5-824a4cbfe531)

Finally, I can apply the `$3.1 M` per WAR to the predicted WAR's to see a valid contract put together. I will get the AAV (Average Annual Value) breakdown for what the player's value should be in each of their future aged seasons.

![image](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/51abe9da-4c3b-441d-987b-12ea018c7fd7)

This walkthrough can now be applied to any player with data from the 2022 season! I used Trea Turner as my example, but the mechanics behind it all are the same. The code for the cells described can be found in `offensive-predictor.ipynb`. As I move forward, I am now going to look for ways to fine tune the regression techniques, and research other factors that could be implemented when creating these contracts.

## Tuning the Model and Researching Contracts
If I already previously haven't explained WAR, I want to reiterate that it is a singular stat that tries to quantify a player's value, but is not necessarily a perfectly precise indicator of player's contribution. So when I run my contract predictor as I constructed it above, it provides a baseline of what player's should be paid, when in reality looking at contracts that were signed this past offseason, players were getting more money. Although contracts are a negotiation in most cases, so I am going to try to find something to help take the contract evaluator to the next level.

I began to investigate the relationship that I found using WAR, and decided to see how WAR is calculated. I got my WAR values from Baseball Reference, but FanGraphs actually calculates their stat `fWAR` slightly differently. Interestingly, when I use the `fWAR` value from FanGraphs with the `WAR` relationship found by my model, the contracts are actually very close to what player's signed for this past offseason! The results are actually quite impressive in my opinion. Below I will show some of the contracts with my predicted contract from running it through the predictor, vs what they actually signed for:

![download](https://github.com/eackerm2/MLB-Contract-AI-ML-Project/assets/122949257/973ce94d-6a1d-4053-95ac-77e845daa787)





