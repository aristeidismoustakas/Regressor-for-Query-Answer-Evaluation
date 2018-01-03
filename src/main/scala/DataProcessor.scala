import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.collect_list
import org.apache.spark.ml.feature.StopWordsRemover
//import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.ml.feature.Tokenizer


object DataProcessor {

    def main(args: Array[String]): Unit = {
        val df_tup = read_data("data/train.csv", "data/attributes.csv", "data/product_descriptions.csv")
        df_tup._1.show(10)
        df_tup._2.show(10)
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
    def read_data(train_csv: String, attributes_csv: String, descriptions_csv: String): (DataFrame, DataFrame) = {
        val spark = SparkSession
            .builder()
            .master("local")
            .appName("RelevancePrediction")
            .getOrCreate()

        // Read train.csv
        val train_df = spark.read.format("csv").option("header", "true").csv(train_csv)

        // Read attributes.csv, group by product and collect all attributes per product in a list
        val attributes_df = spark.read.format("csv").option("header", "true").csv(attributes_csv)
        val attributes_df_grouped = attributes_df.groupBy(attributes_df.col("product_uid")).agg(collect_list(columnName = "value"))

        // Read product_descriptions.csv
        val descriptions_df = spark.read.format("csv").option("header", "true").csv(descriptions_csv)

        // Separate product titles from training dataframe
        val train_df_without_product_names = train_df.select("id", "product_uid", "search_term", "relevance")
        val train_df_product_names = train_df.select("product_uid", "product_title").dropDuplicates("product_uid")

        // Create a dataframe with all product info
        val products = descriptions_df
            .join(attributes_df_grouped, Seq("product_uid"), "outer")
            .join(train_df_product_names, Seq("product_uid"), "outer")

        // Return
        (train_df_without_product_names, products.toDF("product_uid", "product_description", "product_attributes", "product_title"))

        removeStopWords(train_df_without_product_names.toDF(),"product_description")
    }

    def removeStopWords(train_df: DataFrame,colToCl:String): Unit = {
        val remover = new StopWordsRemover()
          .setInputCol(colToCl)
          .setOutputCol(colToCl+"_cl")

        remover.transform(train_df).show(false)
        train_df.head(4).foreach(println)
    }


}
