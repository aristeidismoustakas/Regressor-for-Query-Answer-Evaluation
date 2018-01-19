name := "RelevancePredictionForQueryAnswerEvaluation"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"

resolvers ++= Seq("spark-stemming" at "https://dl.bintray.com/spark-packages/maven/")

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "master" % "spark-stemming" % "0.2.0",

    "org.scalanlp" %% "breeze" % "0.13.2",

    // Native libraries are not included by default. add this if you want them (as of 0.7)
    // Native libraries greatly improve performance, but increase jar sizes.
    // It also packages various blas implementations, which have licenses that may or may not
    // be compatible with the Apache License. No GPL code, as best I know.
    "org.scalanlp" %% "breeze-natives" % "0.13.2",

    // The visualization library is distributed separately as well.
    // It depends on LGPL code
    "org.scalanlp" %% "breeze-viz" % "0.13.2"

)

//libraryDependencies += "com.github.master" %% "spark-stemming" % "0.2.0"