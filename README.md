# Predicting-click-through-rate-of-ads
Taken the dataset from KDD 2012 cup which is of around 10gb. We have placed the training file on hadoop cluster.  Initially, we have used pig to transform the data. Later, we have used Spark MLlib for dimensionality reduction and model building. Finally, evaluated the models using some evaluation metrics.

Step1: Place the given training sample file on cluster

Step2: update the directory in all .scala files 

step3: put all .scala files in your local directory on local server.

Step4: Go to spark shell by command 'spark-shell'

step5: Execute the .scala files by following commands

:load linregwithpca.scala
:load linregwithoutpca.scala
:load decisiontreewithpca.scala
:load decisiontreewithoutpca.scala
:load randomforestwithpca.scala
:load randomforestwithoutpca.scala
:load boostingwithpca.scala
:load boostingwithoutpca.scala

