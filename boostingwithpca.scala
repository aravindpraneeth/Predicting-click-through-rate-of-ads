
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

val data1=sc.textFile("hadoop directory").map(line=>line.split("\\t").map(_.toDouble)).map(line=>(line(2),line(3),line(4),line(5),line(6),line(7),line(8),line(9),line(10),line(11),(line(0)/line(1))))

val parsedData = data1.map { parts =>LabeledPoint(parts._11.toDouble, Vectors.dense(parts._1.toDouble,parts._2.toDouble,parts._3.toDouble,parts._4.toDouble,parts._5.toDouble,parts._6.toDouble,parts._7.toDouble,parts._8.toDouble,parts._9.toDouble,parts._10.toDouble))}.cache()

val pca = new PCA(8).fit(parsedData.map(_.features))

val projected = parsedData.map(p => p.copy(features = pca.transform(p.features)))
  
val splits = projected.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
val boostingStrategy = BoostingStrategy.defaultParams("Regression")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.maxDepth = 5
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(training, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelsAndPredictions = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testMSEboost = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSEboost)
