import org.apache.spark.sql.{DataFrame, SparkSession}

object DataProcessor {

    def main(args: Array[String]): Unit = {
        val df_tup = read_data("data/train.csv", "data/attributes.csv", "data/product_descriptions.csv")

        df_tup._1.show(10)
        df_tup._2.show(10)
        df_tup._3.show(10)
    }


    def read_data(train_csv: String, attributes_csv: String, descriptions_csv: String): (DataFrame, DataFrame, DataFrame) = {
        val spark = SparkSession
            .builder()
            .master("local")
            .appName("RelevancePrediction")
            .getOrCreate()

        val train_df = spark.read.format("csv").option("header", "true").csv(train_csv)
        val attributes_df = spark.read.format("csv").option("header", "true").csv(attributes_csv)
        val descriptions_df = spark.read.format("csv").option("header", "true").csv(descriptions_csv)

        (train_df, attributes_df, descriptions_df)
    }

    def process_data(train_df: DataFrame, attributes_df: DataFrame, descriptions_df: DataFrame): Unit = {

    }

}
