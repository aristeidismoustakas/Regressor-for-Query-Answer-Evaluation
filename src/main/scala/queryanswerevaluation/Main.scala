package queryanswerevaluation

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, LinearRegression}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.sql.functions.udf
import queryanswerevaluation.DataProcessor.{process_data, read_data}
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.io._

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

object Main {

    // Helper function to delete directory if it exists
    def delete(dir: String): Unit = {
        val file = new File(dir)
        if (file.isDirectory)
            file.listFiles().foreach(f => delete(f.getAbsolutePath))
        if (file.exists && !file.delete)
            throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
    }

    def main(args: Array[String]): Unit = {
        val conf = new SparkConf()
            .setAppName("RelevancePrediction")
            .setMaster("local[4]")
            .set("spark.executor.memory", "1g")
        val spark = SparkSession.builder().config(conf).getOrCreate()

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val train_col_names = Seq("product_uid", "id", "product_title",
            "search_term", "relevance", "product_attributes",
            "product_description")

        val test_col_names = Seq("product_uid", "id", "product_title",
            "search_term", "product_attributes", "product_description")

        val toDouble = udf[Double, String]( _.toDouble)

        // Read training set
        val train_rel_str = read_data("data/train.csv", "data/attributes.csv", "data/product_descriptions.csv")
            .toDF(train_col_names: _*)

        // Cast relevance to double
        val train = train_rel_str
            .withColumn("rel_num", toDouble(train_rel_str.col("relevance")))
            .drop("relevance")
            .withColumnRenamed("rel_num", "relevance")

        // Read test set
        val test = read_data("data/test.csv", "data/attributes.csv", "data/product_descriptions.csv")
            .toDF(test_col_names:_*)

        // Process training set
        val processed_train_df = process_data(train)
        processed_train_df.persist(StorageLevel.MEMORY_ONLY)
        processed_train_df.show(20)

        // Process test set
        val processed_test_df = process_data(test)

        // Split into training and validation set - 70%/30%
        val split = processed_train_df.randomSplit(Array(0.7, 0.3), seed=42)

        val training_set = split(0)
        val validation_set = split(1)

        val linear_regression_results = train_linear_regression(training_set, validation_set, "ltr_features", "relevance")
        println("Linear Regression: RMSE: " + linear_regression_results._2 + " Time: " + linear_regression_results._3 + "s")

        val regressor_tree_results = train_regressor_tree(training_set, validation_set, "ltr_features", "relevance")
        println("Regressor Tree: RMSE: " + regressor_tree_results._2 + " Time: " + regressor_tree_results._3 + "s")

        val gb_trees_results = train_gradient_boosted_trees(training_set, validation_set, "ltr_features", "relevance")
        println("Gradient Boosted Trees: RMSE: " + gb_trees_results._2 + " Time: " + gb_trees_results._3 + "s")

        // Write results on true test set
//        delete("results")
//        regressor_tree_results._1.transform(processed_test_df).select("id", "prediction").write.csv("results")
    }

    def train_linear_regression(training_set: DataFrame, test_set: DataFrame, features: String, labels: String) = {
        val t0 = System.currentTimeMillis()

        // Train model
        val rf = new LinearRegression()
            .setLabelCol(labels)
            .setFeaturesCol(features)
            .setMaxIter(10)
            .fit(training_set)

        // Get predictions
        val predictions = rf.transform(test_set)

        // Evaluate model
        val evaluator = new RegressionEvaluator()
            .setLabelCol("relevance")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)

        val t1 = System.currentTimeMillis()

        (rf, rmse, (t1-t0)/1000)
    }

    def train_regressor_tree(training_set: DataFrame, test_set: DataFrame, features: String, labels: String) = {
        val t0 = System.currentTimeMillis()

        // Train model
        val rf = new DecisionTreeRegressor()
            .setLabelCol(labels)
            .setFeaturesCol(features)
            .fit(training_set)

        // Get predictions
        val predictions = rf.transform(test_set)

        // Evaluate model
        val evaluator = new RegressionEvaluator()
            .setLabelCol("relevance")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)

        val t1 = System.currentTimeMillis()

        (rf, rmse, (t1-t0)/1000)
    }

    def train_gradient_boosted_trees(training_set: DataFrame, test_set: DataFrame, features: String, labels: String) = {
        val t0 = System.currentTimeMillis()

        // Train model
        val rf = new GBTRegressor()
            .setLabelCol(labels)
            .setFeaturesCol(features)
            .fit(training_set)

        // Get predictions
        val predictions = rf.transform(test_set)

        // Evaluate model
        val evaluator = new RegressionEvaluator()
            .setLabelCol("relevance")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)

        val t1 = System.currentTimeMillis()

        (rf, rmse, (t1-t0)/1000)
    }

}
