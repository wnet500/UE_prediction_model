from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


#############################################################################
# calibraion plot and brier score for theb best model using rf
probas_rf = clf_rf.predict_proba(X_test_selected)

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probas_rf, n_bins=10)
clf_score = brier_score_loss(y_test, probas_rf, pos_label=y_test.max())

## plotting
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams.update({'axes.labelsize': 'large'})

plt.plot([0, 1], [0, 1],"k:", label="Perfectly calibrated")
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=r"Random Forest (Brier score: %0.3f)" % (clf_score))

plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.legend(loc="lower right")

plt.savefig("rf_calibration_plot", dpi=500)

plt.show()