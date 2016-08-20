
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors

val data1=sc.textFile("hadoop directory").map(line=>line.split("\\t").map(_.toDouble)).map(line=>(line(2),line(3),line(4),line(5),line(6),line(7),line(8),line(9),line(10),line(11),(line(0)/line(1))))

val parsedData = data1.map { parts =>LabeledPoint(parts._11.toDouble, Vectors.dense(parts._1.toDouble,parts._2.toDouble,parts._3.toDouble,parts._4.toDouble,parts._5.toDouble,parts._6.toDouble,parts._7.toDouble,parts._8.toDouble,parts._9.toDouble,parts._10.toDouble))}.cache()
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainRegressor(training, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelsAndPredictions = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)
println("Learned regression forest model:\n" + model.toDebugString)

