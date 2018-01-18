import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.linalg.functions.cosineDistance
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.RandomForestRegressor

object DataProcessor {

    def main(args: Array[String]): Unit = {
        val t0 = System.currentTimeMillis()

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val toDouble = udf[Double, String]( _.toDouble)

        val train_rel_str = read_data("data/train.csv", "data/attributes.csv", "data/product_descriptions.csv").toDF("product_uid", "id", "product_title",
            "search_term", "relevance", "product_attributes", "product_description")

        val train = train_rel_str.withColumn("rel_num", toDouble(train_rel_str.col("relevance"))).drop("relevance").withColumnRenamed("rel_num", "relevance")
//
//        val test = read_data("data/test.csv", "data/attributes.csv", "data/product_descriptions.csv").toDF("product_uid", "id", "product_title",
//            "search_term", "product_attributes", "product_description")

        val processed_train_df = process_data(train)
//        val processed_test_df = process_data(test)

        processed_train_df.show(10)
//        processed_test_df.show(10)

        val split = processed_train_df.randomSplit(Array(0.7, 0.3), seed=42)

        val training_set = split(0)
        val validation_set = split(1)

        val rf = new RandomForestRegressor()
            .setLabelCol("relevance")
            .setFeaturesCol("features")
            .fit(training_set)

        val predictions = rf.transform(validation_set)

        val evaluator = new RegressionEvaluator()
            .setLabelCol("relevance")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)

        val t1 = System.currentTimeMillis()

        println("Execution time: " + (t1-t0)/1000 + "s")
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    }

    /***
      * Reads data from CSV files and returns 2 DataFrames
      * First DataFrame contains: (id, product_uid, search_term, relevance)
      * Second DataFrame contains: (product_uid, product_description, product_attributes, product_title)
      * @param train_csv train.csv
      * @param attributes_csv attributes.csv
      * @param descriptions_csv product_descriptions.csv
      * @return Tuple with the 2 DataFrames
      */
    def read_data(train_csv: String, attributes_csv: String, descriptions_csv: String): DataFrame = {
        val spark = SparkSession
            .builder()
            .master("local")
            .appName("RelevancePrediction")
            .getOrCreate()

        val train_df = spark.read.format("csv").option("header", "true").csv(train_csv)

        val attributes_df = spark.read.format("csv").option("header", "true").csv(attributes_csv)
        val attributes_df_grouped = attributes_df.groupBy(attributes_df.col("product_uid")).agg(concat_ws(" ", collect_list(columnName = "value")))
        val descriptions_df = spark.read.format("csv").option("header", "true").csv(descriptions_csv)
//
//        val train_df_without_product_names = train_df.select("id", "product_uid", "search_term", "relevance")
//        val train_df_product_names = train_df.select("product_uid", "product_title").dropDuplicates("product_uid")

        val train = train_df
            .join(attributes_df_grouped, Seq("product_uid"), "left_outer")
            .join(descriptions_df, Seq("product_uid"), "left_outer")
            .na.fill("")

        train
    }

    def process_data(products: DataFrame): DataFrame = {
        val col_names = Seq("search_term", "product_description", "product_attributes", "product_title")
        var dfs = new Array[DataFrame](col_names.length + 1)

        val udf_cos_sim = udf((vec1: DenseVector, vec2: DenseVector) => {
            val vec_1_bdv = BDV(vec1.toDense.toArray)
            val vec_2_bdv = BDV(vec2.toDense.toArray)

            1 - cosineDistance(vec_1_bdv, vec_2_bdv)
        })

        dfs(0) = products
        var count = 0

        val udf_toDense = udf((v: SparseVector) => v.toDense)

        col_names.foreach(col_name => {
            val tokenizer = new Tokenizer()
                .setInputCol(col_name)
                .setOutputCol(col_name+"_tokenized")

            val remover = new StopWordsRemover()
                .setInputCol(col_name+"_tokenized")
                .setOutputCol(col_name+"_clean")

            val hash_tf = new HashingTF()
                .setNumFeatures(20000)
                .setInputCol(col_name+"_clean")
                .setOutputCol(col_name+"_tf")

            val idf = new IDF()
              .setInputCol(col_name+"_tf")
              .setOutputCol(col_name+"_tfidf_sparse")

            val tokenized = tokenizer.transform(dfs(count))
            val clean = remover.transform(tokenized)
            val tf = hash_tf.transform(clean)
            val idf_fit = idf.fit(tf)
            val processed_col_df = idf_fit.transform(tf)
                .drop(col_name)
                .drop(col_name+"_tokenized")
                .drop(col_name+"_clean")
                .drop(col_name+"_tf")


            count = count + 1
            dfs(count) = processed_col_df
                .withColumn(col_name+"_tfidf", udf_toDense(processed_col_df.col(col_name+"_tfidf_sparse")))
                .drop(col_name+"_tfidf_sparse")
        })

        // Return processed dataset
        val tfidf = dfs(count)
        val sims = tfidf
            .withColumn("search_term_title_cos_sim", udf_cos_sim(tfidf.col("search_term_tfidf"), tfidf.col("product_title_tfidf")))
            .withColumn("search_term_desc_cos_sim", udf_cos_sim(tfidf.col("search_term_tfidf"), tfidf.col("product_description_tfidf")))
            .withColumn("search_term_attr_cos_sim", udf_cos_sim(tfidf.col("search_term_tfidf"), tfidf.col("product_attributes_tfidf")))

        val assembler = new VectorAssembler()
            .setInputCols(Array("search_term_title_cos_sim", "search_term_desc_cos_sim", "search_term_attr_cos_sim"))
            .setOutputCol("features")

        assembler.transform(sims)
    }

}
