#############################################################################
# decision curve

def Netbenefit(threshold_of_interests, y_probas, y_test, decimals=5):
    
    NBs = []
    Alls = []
    
    for threshold_of_interest in threshold_of_interests:

        y_pred = y_probas >= threshold_of_interest

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        N = len(y_test)
        w = threshold_of_interest / (1-threshold_of_interest)
        NB = tp/N - w*fp/N
        
        All = (y_test.value_counts()[1] / N) - (w * y_test.value_counts()[0]/N)
        
        NBs.append(np.around(NB, decimals=decimals))
        Alls.append(np.around(All, decimals=decimals))
        
    return(NBs, Alls)

threshold_of_interests = [i for i in np.arange(0, 1, 0.001)]

y_probas = clf_rf.predict_proba(X_test_selected)[:, 1]
NBs, Alls = Netbenefit(threshold_of_interests, y_probas, y_test)

## plotting
plt.plot(threshold_of_interests, NBs, color="red", label="UE prediction model")
plt.plot(threshold_of_interests, Alls, color="blue", label="Predicting all as an UE")
plt.axhline(y=0, color='k', label="Predicting none as an UE")
plt.axvline(x=0.142, color='green', linestyle='--', label="Selected threshold")

plt.ylim([-0.1, 0.25])

#plt.title("Decision curve\n")
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.legend(loc="uper left")
plt.rc('grid', linestyle="--", color='silver')
plt.grid(True)
plt.savefig('Net_Benefit', dpi=500)
plt.show()

