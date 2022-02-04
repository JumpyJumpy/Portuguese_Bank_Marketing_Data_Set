library(tidyverse)
library(MASS)
library(ggplot2)
library(car)
library(hrbrthemes)


bank <- read.csv("./data/bank.csv", stringsAsFactors = T)

str(bank)

num_var <- c(
    'age',
    'default',
    'housing',
    'loan',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed'
)

cate_var <-
    c('job',
      'marital',
      'education',
      'contact',
      'month',
      'day_of_week')

bank$y[bank$y == 0] = "no"
bank$y[bank$y == 1] = "yes"

for (var in num_var) {
    ggplot(data = bank) +
        geom_jitter(aes_string(x = var, y = "y", color = "y"), height = 0.1, size = 0.01) +
        labs(title = paste0(var, " vs. y"),
             x = var,
             y = "y") +
        theme(legend.position = "none") + facet_grid()
}

y_jitter <- jitter(bank$y)


ggplot() + 
    geom_jitter(aes(x = , y = ), height = 0.1, size = 0.1)
