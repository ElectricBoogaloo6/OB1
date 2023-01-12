library(lme4)
library(ggplot2)
library(plyr)

setwd('C://Users//Josh//Desktop//josh work//Experiments//BOB//sam reading model july15//sam reading model july15//Raw_BT_data')
data <- read.csv('data.txt')

data <- data[data$condition == 2,]

mean(data$refix)
mean(data$activity)

