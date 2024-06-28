# PURPOSE: Plot DICE score distributions

# Load libraries
library(gplots)
library(ggplot2)
library("tidyverse")

############ LOAD THE FOUR SPREADSHEETS WE ARE GIVEN ############
set.seed(0)
homedir <- "/Users/Kalina/Documents/CBIG/Project_MedSAM_Testing/Results/"
dice_val <- read.csv(paste(homedir,"baselines_val/dice_scores_baseline_val.csv", sep = ""))
dice_train <- read.csv(paste(homedir,"baselines_train/dice_scores_baseline_train.csv", sep = ""))

# Remove any NaNs
dice_val <- na.omit(dice_val) # remove cases with any NaNs
dice_train <- na.omit(dice_train) # remove cases with any NaNs

# Rename columns
colnames(dice_val)[1] <- "File_Path"
colnames(dice_val)[2] <- "DICE"
colnames(dice_train)[1] <- "File_Path"
colnames(dice_train)[2] <- "DICE"

# Plot the DICE scores as density plot

ggplot(dice_val, aes(DICE)) + geom_density(alpha = 0.2)


# Plot the DICE scores as a simple histogram
plot_title_val <- "Baseline DICE scores in validation set"
plot_title_train <- "Baseline DICE scores in training set"
x_lab <- "DICE"
hist(dice_val$DICE,main = plot_title_val ,xlab=x_lab, freq=FALSE)
hist(dice_train$DICE,main = plot_title_train ,xlab=x_lab, freq=FALSE)