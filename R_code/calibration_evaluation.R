
# The Integrated Calibration Index (ICI) ----------------------------------
# predicted_probas indicate "a vector of predicted probabilities by the model"
# observed_binary_outcomes indicate "a vector of observed binary outcomes"

data <- data.frame(predictions = predicted_probas, observations = observed_binary_outcomes)

loess.calibrate <- loess(data$obervations ~ data$predictions)

P.calibrate <- predict(loess.calibrate, newdata = data$predictions)

mean(abs(P.calibrate - data$predictions))


# Hosmer-Lemeshow goodness-of-fit statistic -------------------------------
library(ResourceSelection)

hoslem.test(data$observations, data$predictions)
