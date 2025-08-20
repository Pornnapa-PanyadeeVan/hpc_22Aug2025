# regression_example.R

# Linear Regression R


# dataset 

height <- c(150, 155, 160, 165, 170, 175, 180)

weight <- c(50, 55, 58, 60, 65, 70, 72)



data <- data.frame(height, weight)




print("dataset:")

print(data)



# Linear Model (weight ~ height)

model <- lm(weight ~ height, data = data)



# regression

print("Summary:")

print(summary(model))



# model 

new_heights <- data.frame(height = c(162, 168, 182))

predictions <- predict(model, newdata = new_heights)



print("Prediction:")

print(data.frame(new_heights, predicted_weight = predictions))



# regression
plot(data$height, data$weight,
     main="Regression: Weight ~ Height",
     xlab="Height (cm)", ylab="Weight (kg)", 
     pch=19, col="blue")

abline(model, col="red", lwd=2)

