name := "RelevancePredictionForQueryAnswerEvaluation"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion
    //"com.github.master" %% "spark-stemming" % sparkVersion
)

//libraryDependencies += "com.github.master" %% "spark-stemming" % "0.2.0"