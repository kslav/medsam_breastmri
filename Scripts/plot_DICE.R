# PURPOSE: Plot DICE score distributions

# Load libraries
library(gplots)
library(ggplot2)
library("tidyverse")

############ LOAD THE FOUR SPREADSHEETS WE ARE GIVEN ############
set.seed(0)
homedir <- "/Users/Kalina/Documents/CBIG/Project_MedSAM_Testing/Results/"
dice_val <- read.csv(paste(homedir,"dice_scores_mean.csv", sep = ""))
#dice_train <- read.csv(paste(homedir,"baselines_train/dice_scores_baseline_train.csv", sep = ""))

# Remove any NaNs
#dice_val <- na.omit(dice_val) # remove cases with any NaNs
#dice_train <- na.omit(dice_train) # remove cases with any NaNs

# Rename columns
#colnames(dice_val)[1] <- "File_Path"
#colnames(dice_val)[2] <- "DICE"
#colnames(dice_train)[1] <- "File_Path"
#colnames(dice_train)[2] <- "DICE"

# Plot the DICE scores as a simple histogram
plot_title_val <- "Baseline Dice Similarity Coefficient Scores"
#plot_title_train <- "Baseline DICE scores in training set"
x_lab <- "Dice Similarity Coefficient"
hist(dice_val$DICE,main = plot_title_val ,xlab=x_lab, freq=TRUE)
abline(v=mean(dice_val$DICE), col='red', lwd=3, lty='dashed')
hist(dice_val$Size,main = "Lesion sizes" ,xlab="Number of pixels", freq=TRUE)
#hist(dice_train$DICE,main = plot_title_train ,xlab=x_lab, freq=FALSE)
# get the quantiles of the tumor sizes
res<-quantile(dice_val$Size, probs = c(0,0.25,0.5,0.75,1)) 
iqr <- IQR(dice_val$Size)

thresh <- res[4]+1.5*iqr

# what does dice as a funciton of size look like?
ggplot(dice_val, aes(x=Size,y=DICE)) + geom_point() + geom_vline(xintercept = median(dice_val$Size), linetype="dotted", 
                                                                  color = "blue", size=1.5)
ggplot(dice_val[dice_val$Size<100000,], aes(x=Size,y=DICE)) + geom_point() + geom_vline(xintercept = thresh, linetype="dotted", 
                                                                                      color = "blue", size=1.5)

without_outliers <-dice_val[dice_val$Size<thresh,] #265 non-outliers

# compute correlation coefficient for kicks
corr_s <- cor.test(without_outliers$DICE,without_outliers$Size, method = 'spearman',exact=FALSE)
pval_s <- corr_s$pval
corr_p <- cor.test(without_outliers$DICE,without_outliers$Size, method = 'pearson',exact=FALSE)
pval_p <- corr_p$pval


ggplot(dice_val[dice_val$Size<thresh,], aes(x=Size,y=DICE)) + geom_point() + geom_vline(xintercept = median(dice_val$Size), linetype="dotted", 
                                                                                        color = "blue", size=1.5)

hist(dice_val$Size[dice_val$Size<thresh], main="Lesion sizes excluding outliers",xlab="Number of pixels")