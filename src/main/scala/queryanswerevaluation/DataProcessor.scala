package queryanswerevaluation

import breeze.linalg.functions.{cosineDistance, euclideanDistance, tanimotoDistance}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataProcessor {
    // User Defined Function to calculate ltr_features between two columns
    val udf_ltr = udf((vec1: DenseVector, vec2: DenseVector) => {
        val vec_1_bdv = BDV(vec1.toDense.toArray)
        val vec_2_bdv = BDV(vec2.toDense.toArray)

        val sims = List(1 - cosineDistance(vec_1_bdv, vec_2_bdv),
            1 - euclideanDistance(vec_1_bdv, vec_2_bdv),
            1 - tanimotoDistance(vec_1_bdv, vec_2_bdv))

        Vectors.dense(sims.toArray)
    })

    /***
      * Reads data from CSV files and returns 2 DataFrames
      * First DataFrame contains: (id, product_uid, search_term, relevance)
      * Second DataFrame contains: (product_uid, product_description, product_attributes, product_title)
      * @param csv_file train.csv
      * @param attributes_csv attributes.csv
      * @param descriptions_csv product_descriptions.csv
      * @return Tuple with the 2 DataFrames
      */
    def read_data(csv_file: String, attributes_csv: String, descriptions_csv: String): DataFrame = {
        val spark = SparkSession
            .builder()
            .master("local")
            .appName("RelevancePrediction")
            .getOrCreate()

        val csv_df = spark.read.format("csv").option("header", "true").csv(csv_file)

        val attributes_df = spark.read.format("csv").option("header", "true").csv(attributes_csv)
        val attributes_df_grouped = attributes_df.groupBy(attributes_df.col("product_uid")).agg(concat_ws(" ", collect_list(columnName = "value")))
        val descriptions_df = spark.read.format("csv").option("header", "true").csv(descriptions_csv)

        val final_df = csv_df
            .join(attributes_df_grouped, Seq("product_uid"), "left_outer")
            .join(descriptions_df, Seq("product_uid"), "left_outer")
            .na.fill("")

        final_df
    }

    def process_data(products: DataFrame): DataFrame = {
        val col_names = Seq("search_term", "product_description", "product_attributes", "product_title")
        // Array to keep dataframes after each iteration
        var dfs = new Array[DataFrame](col_names.length + 1)

        // First df that will be processed
        dfs(0) = products
        var count = 0

        // User Defined Function to cast sparse to dense vectors
        val udf_toDense = udf((v: SparseVector) => v.toDense)

        col_names.foreach(col_name => {
            // For each column, tokenize, remove stop words, stem and create tfidf vectors
            val tokenizer = new Tokenizer()
                .setInputCol(col_name)
                .setOutputCol(col_name+"_tokenized")

            val remover = new StopWordsRemover()
                .setInputCol(tokenizer.getOutputCol)
                .setOutputCol(col_name+"_clean")

            val stemmer = new Stemmer()
                .setInputCol(remover.getOutputCol)
                .setOutputCol(col_name+"_stemmed")

            val hash_tf = new HashingTF()
                .setNumFeatures(20000)
                .setInputCol(stemmer.getOutputCol)
                .setOutputCol(col_name+"_tf")

            val idf = new IDF()
                .setInputCol(hash_tf.getOutputCol)
                .setOutputCol(col_name+"_tfidf_sparse")

            val tokenized = tokenizer.transform(dfs(count))
            val clean = remover.transform(tokenized)
            val stemmed = stemmer.transform(clean)
            val tf = hash_tf.transform(stemmed)
            val idf_fit = idf.fit(tf)
            val processed_col_df = idf_fit.transform(tf)
                .drop(col_name)
                .drop(col_name+"_tokenized")
                .drop(col_name+"_clean")
                .drop(col_name+"_stemmed")
                .drop(col_name+"_tf")

            // Increment count and store current df in array after casting vectors to dense
            count = count + 1
            dfs(count) = processed_col_df
                .withColumn(col_name+"_tfidf", udf_toDense(processed_col_df.col(col_name+"_tfidf_sparse")))
                .drop(col_name+"_tfidf_sparse")
        })

        // Get latest dataframe, with all columns processed
        val tfidf = dfs(count)
        // Calculate similarities
        val sims = tfidf
            .withColumn("search_term_title_sims", udf_ltr(tfidf.col("search_term_tfidf"), tfidf.col("product_title_tfidf")))
            .withColumn("search_term_desc_sims", udf_ltr(tfidf.col("search_term_tfidf"), tfidf.col("product_description_tfidf")))
            .withColumn("search_term_attr_sims", udf_ltr(tfidf.col("search_term_tfidf"), tfidf.col("product_attributes_tfidf")))

        // Assemble similarities in vector
        val assembler = new VectorAssembler()
            .setInputCols(Array("search_term_title_sims", "search_term_desc_sims", "search_term_attr_sims"))
            .setOutputCol("ltr_features")

        // Transform and return result
        assembler.transform(sims)
    }

}
