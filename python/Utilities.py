
import datetime
import seaborn as sns
import pandas as pd
import numpy as np
import time

class Utility():

    starttime=None
    def startTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This must be the first method to call before calling stopTimeTrack or checkpointTimeTrack
        Will start the recording of time.
        :return:
        """
        self.starttime=datetime.datetime.now()
        pass

    def stopTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This is called in pair with startTimeTrack everytime.
        It will print time lapse after startTimeTrack
        :return:
        """
        endtime=datetime.datetime.now()
        duration=endtime-self.starttime
        result="Time taken: ",duration.seconds," secs"
        print(result)
        return result
        pass


    def checkpointTimeTrack(self):
        """
        Changelog: 
        - 29/03 KS First committed        
        This can be called consecutively for as many times as long as startTimeTrack has been first called.
        It will print the time lapse from last check point
        E.g.
        Utility().startTimeTrack)_
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        Utility().checkpointTimeTrack()
        :return:
        """
        endtime = datetime.datetime.now()
        duration = endtime - self.starttime
        result=""
        if(duration.seconds>60):
            result = "Time taken: ", duration.seconds/60, " mins"
        else:
            result = "Time taken: ", duration.seconds, " secs"
        self.starttime=endtime
        print(result)
        return result
        pass


    def correlationFeatures(self,dataset):
        print(list(dataset))

        # dataset=dataset.filter(items=['click', 'weekday', 'hour', 'userid', 'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage', 'advertiser', 'os', 'browser'])
        corr=dataset.corr()
        print(corr)
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        sns.plt.show()

    def artificialFeatureExtension(self,featureDF):
        def multiply(x,f1='',f2=''):
            return x[f1]*x[f2]

        listOfFeatures=list(featureDF)
        noOfFeatures=len(listOfFeatures)
        i=0
        while(i<noOfFeatures):
            subFeatures=noOfFeatures-i

            j = 0
            while(j<subFeatures):
                f1=listOfFeatures[i]
                f2=listOfFeatures[i+1]
                featureName=str(f1+"_"+f2)
                print("I is :, ", i, "    No of features: ", subFeatures)
                print("J is :, ", j, "    No of subfeatures: ", subFeatures)
                print("Working on features: ", featureName)
                featureDF[featureName]=featureDF.apply(multiply,axis=1,f1=f1, f2=f2)
                j=j+1
                time.sleep(0.5)
            i=i+1
        return featureDF

# print("Reading feature list")
# all_df=pd.read_csv('../data/features_full_plusnouns_pluspuidthresh.csv')
# feature_train_df = all_df[:74067]
# featureDF=Utility().artificialFeatureExtension(feature_train_df)
# featureDF.to_csv('nonlinear_features')
# print("Created featureDF: ", list(featureDF))
