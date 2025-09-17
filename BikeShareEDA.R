library(tidyverse)
library(tidyr)
library(vroom)
library(ggplot2)
library(patchwork)

#Load Data to df
train_data <- vroom("data/train.csv")
test_data <- vroom("data/test.csv")


##EDA##
ggplot(data = train_data ,aes(x = count)) + geom_histogram(binwidth = 50)

ggplot(data = train_data ,aes(x = datetime ,y = humidity)) + geom_point() + geom_smooth()

ggplot(data = train_data ,aes(x = datetime ,y = windspeed)) + 
    geom_point() + 
    geom_smooth()

ggplot(data = train_data ,aes(x = datetime ,y = atemp)) + 
    geom_point() + 
    geom_smooth()


#4 EDA plots ##

##Count by temp
plot1 <- ggplot(data = train_data ,aes(x = atemp ,y = count)) + 
  geom_point() + 
  geom_smooth() +
  labs (
    title = "Count Across Temperature"
    ,x = "Feels Like Temperature (C)"
    ,y = "Count"
  )

#Count by weather type
plot2 <- ggplot(train_data ,aes(x = factor(weather) ,y =  count)) +
  geom_col(fill = "darkblue") +
  labs (
    title = "Distribution of Counts by Weather Type"
    ,x = "Weather Type"
    ,y = "Count of Bike Shares"
  ) +
  scale_x_discrete(
    labels = c(
      "1" = "Clear / Few clouds",
      "2" = "Misty / Cloudy",
      "3" = "Light Snow / Light Rain",
      "4" = "Heavy Rain / Snow"
    ))

##Counts by day of week
train_data_by_dayoweek <- train_data %>% 
  mutate(weekday = wday(datetime, label = TRUE, abbr = TRUE)) %>%
  group_by(weekday) %>%
  summarise(mean_count = mean(count))

plot3 <- ggplot(train_data_by_dayoweek ,aes(x = weekday ,y = mean_count)) +
  geom_bar(stat = "identity") +
  labs (
    title = "Count of Bike Shares by Day of Week"
    ,x = "Day of Week"
    ,y = "Mean Count of Bike Share"
  )

##Count by hour of day
plot4 <- ggplot(data = train_data ,aes(x = hour(datetime) ,y = count)) +
    geom_histogram(stat = "identity") +
    labs(
      title = "Count by Hour of Day"
      ,x = "Hour (0-24)"
      ,y = "Count of Bike Shares"
    )

#create 4 panel plot
(plot1 + plot2) / (plot3 + plot4) #4 panel plot

##Linear Regression##

library(tidymodels)

my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>%
  fit(formula=count~.- datetime, data=train_data)
  
  bike_predictions <- predict(my_linear_model,
                              new_data=test_data)

bike_predictions ## Look at the output

linear_regression_kaggle_submission <- bike_predictions %>%
    bind_cols(. ,test_data) %>%
    select(datetime ,.pred) %>%
    rename(count = .pred) %>%
    mutate(count = pmax(0 ,count)) %>%
    mutate(datetime = as.character(format(datetime)))

vroom_write(x = linear_regression_kaggle_submission ,file = "./LinearPredictions.csv" ,delim = ",")
