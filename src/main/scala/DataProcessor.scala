import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer

import org.apache.log4j.Logger
import org.apache.log4j.Level

object DataProcessor {

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val train = read_data("data/train.csv", "data/attributes.csv", "data/product_descriptions.csv")
        train.show(10)

        val processed_df = process_data(train)
        processed_df.show(10)
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

        train.toDF("product_uid", "id", "product_title",
            "search_term", "relevance", "product_attributes", "product_description")
    }

    def process_data(products: DataFrame): DataFrame = {
        val col_names = Seq("search_term", "product_description", "product_attributes", "product_title")
        var dfs = new Array[DataFrame](col_names.length + 1)

        dfs(0) = products
        var count = 0

        col_names.foreach(col_name => {
            val tokenizer = new Tokenizer()
              .setInputCol(col_name)
              .setOutputCol(col_name+"_tokenized")

            val remover = new StopWordsRemover()
              .setInputCol(col_name+"_tokenized")
              .setOutputCol(col_name+"_clean")

            val hash_tf = new HashingTF()
              .setInputCol(col_name+"_clean")
              .setOutputCol(col_name+"_tf")

            val idf = new IDF()
              .setInputCol(col_name+"_tf")
              .setOutputCol(col_name+"_tfidf")

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
        })

        // Return processed dataset
        dfs(count)
    }

}
