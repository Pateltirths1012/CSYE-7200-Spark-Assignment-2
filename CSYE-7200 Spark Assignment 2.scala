// Databricks notebook source
// MAGIC %scala
// MAGIC import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType};
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.mllib.linalg.Vectors
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC import org.apache.spark.ml.Pipeline
// MAGIC import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
// MAGIC import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
// MAGIC import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
// MAGIC import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassifier

// COMMAND ----------

// 1. Exploratory Data Analysis
// 1.1 Overview
// PassengerId is the unique id of the row and it doesn't have any effect on target
// Survived is the target variable we are trying to predict (0 or 1):
// 1 = Survived
// 0 = Not Survived
// Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
// 1 = Upper Class
// 2 = Middle Class
// 3 = Lower Class
// Name, Sex and Age are self-explanatory
// SibSp is the total number of the passengers' siblings and spouse
// Parch is the total number of the passengers' parents and children
// Ticket is the ticket number of the passenger
// Fare is the passenger fare
// Cabin is the cabin number of the passenger
// Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
// C = Cherbourg
// Q = Queenstown
// S = Southampton
// creating the schema for importing training and testing the datasets


val newTrainSchema = (new StructType)
.add("PassengerId", IntegerType)
.add("Survived", IntegerType)
.add("Pclass", IntegerType)
.add("Name", StringType)
.add("Sex", StringType)
.add("Age", FloatType)
.add("SibSp", IntegerType)
.add("Parch", IntegerType)
.add("Ticket", StringType)
.add("Fare", FloatType)
.add("Cabin", StringType)
.add("Embarked", StringType)

val newTestSchema = (new StructType)
.add("PassengerId", IntegerType)
.add("Pclass", IntegerType)
.add("Name", StringType)
.add("Sex", StringType)
.add("Age", FloatType)
.add("SibSp", IntegerType)
.add("Parch", IntegerType)
.add("Ticket", StringType)
.add("Fare", FloatType)
.add("Cabin", StringType)
.add("Embarked", StringType)

val trainSchema = StructType(newTrainSchema)
val testSchema = StructType(newTestSchema)
val csvFormat = "com.databricks.spark.csv"
val df_train = sqlContext.read.format(csvFormat).option("header","true").schema(trainSchema).load("/FileStore/tables/train-4.csv")
val df_test = sqlContext.read.format(csvFormat).option("header","true").schema(testSchema).load("/FileStore/tables/test-5.csv")

//Creating table views for training and testing
df_train.createOrReplaceTempView("df_train")
df_test.createOrReplaceTempView("df_test")


// COMMAND ----------

//compute numcolumn summary statistics.
df_train.describe("Age", "SibSp", "Parch", "Fare").show()

// COMMAND ----------

df_train.show()

// COMMAND ----------

//Summary stats for cateogrical columns
sqlContext.sql("select Survived, count(*) from df_train group by Survived").show()

// COMMAND ----------

sqlContext.sql("select Pclass, Survived, count(*) from df_train group by Pclass, Survived").show()

// COMMAND ----------

sqlContext.sql("select Sex, Survived, count(*) from df_train group by Sex,Survived").show()

// COMMAND ----------


//Calculating avg Age and Fare to fill null values for training
val AvgAge = df_train.select("Age")
  .agg(avg("Age"))
  .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

// COMMAND ----------

//Calculating the average fare for filling gaps in dataset train
val AvgFare = df_train.select("Fare")
  .agg(avg("Fare"))
  .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

// COMMAND ----------

//Calculate avg Age and Fare to fill null values for test data
val AvgAge_test = df_test.select("Age")
  .agg(avg("Age"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}


// COMMAND ----------

//Calculate average fare for filling gaps in dataset test
val AvgFare_test = df_test.select("Fare")
  .agg(avg("Fare"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}

// COMMAND ----------

// for training
val embarked: (String => String) = {
  case "" => "S"
  case null => "S"
  case a => a
}
val embarkedUDF = udf(embarked)

// COMMAND ----------

//for test
val embarked_test: (String => String) = {
  case "" => "S"
  case null => "S"
  case a => a
}
val embarkedUDF_test = udf(embarked_test)

// COMMAND ----------

//Filling null values with avg values for training dataset
val imputeddf = df_train.na.fill(Map("Fare" -> AvgFare, "Age" -> AvgAge))
val imputeddf2 = imputeddf.withColumn("Embarked", embarkedUDF(imputeddf.col("Embarked")))

//splitting training data into training and validation
val Array(trainingData, validationData) = imputeddf2.randomSplit(Array(0.7, 0.3))

//Filling null values with avg values for test dataset
val imputeddf_test = df_test.na.fill(Map("Fare" -> AvgFare_test, "Age" -> AvgAge_test))
val imputeddf2_test = imputeddf_test.withColumn("Embarked", embarkedUDF_test(imputeddf_test.col("Embarked")))

// COMMAND ----------

// Feature Engineering - Create new attributes that may be derived from the existing attributes. This may include removing certain columns in the dataset. 

//Dropping Cabin feature as it has so many null values
// val df1_train = trainingData.drop("Cabin")

// val df1_test = imputeddf2_test.drop("Cabin")



import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// Assuming you have a DataFrame df_all

// Define the bucketizer splits based on the quantiles
val splits = Array(Double.NegativeInfinity, 7.55, 7.854, 8.05, 10.5, 14.4542, 21.6792, 27.0, 39.6875, 77.9583, 512.329, Double.PositiveInfinity)

// Create a Bucketizer transformer
val bucketizer = new Bucketizer()
  .setInputCol("Fare")  // Specify the input column
  .setOutputCol("FareBucket") // Specify the output column
  .setSplits(splits)

// Apply the bucketizer transformation to the DataFrame
val df_binned = bucketizer.transform(df_train)

// Now df_binned contains the 'FareBucket' column, which represents the binned 'Fare' values


// COMMAND ----------

df_binned.show()

// COMMAND ----------

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

// Assuming you have a DataFrame df_all

// Define the bucketizer splits for quantiles
val splits = Array(Double.NegativeInfinity, 19.0, 22.0, 25.0, 28.0, 31.0, 35.0, 39.0, 45.0, 55.0, 65.0, Double.PositiveInfinity)

// Create a Bucketizer transformer
val bucketizer = new Bucketizer()
  .setInputCol("Age")         // Specify the input column
  .setOutputCol("AgeBucket")  // Specify the output column
  .setSplits(splits)

// Apply the bucketizer transformation to the DataFrame
val df_binned = bucketizer.transform(df_tr)

// Now df_binned contains the 'AgeBucket' column, which represents the binned 'Age' values


// COMMAND ----------

df_binned.show()

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// Assuming you have a DataFrame df_all

// Add a new column 'Family_Size' to df_all
val df_with_family_size: DataFrame = df_binned.withColumn("Family_Size", col("SibSp") + col("Parch") + lit(1))

// Now df_with_family_size contains the 'Family_Size' column


// COMMAND ----------

df_with_family_size.show()

// COMMAND ----------

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// Assuming you have a DataFrame df_all

// Define a Window specification to partition by 'Ticket' column
val windowSpec = Window.partitionBy("Ticket")

// Add a new column 'Ticket_Frequency' to df_all
val df_with_ticket_frequency: DataFrame = df_with_family_size.withColumn("Ticket_Frequency", count("*").over(windowSpec))

// Now df_with_ticket_frequency contains the 'Ticket_Frequency' column

// COMMAND ----------

df_with_ticket_frequency.show()

// COMMAND ----------

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

// Assuming you have a DataFrame df_all
// Initialize a SparkSession
val spark = SparkSession.builder()
  .appName("DataFrame Transformation")
  .getOrCreate()

// Add a new column 'Title' by splitting the 'Name' column
val df_with_title: DataFrame = df_with_ticket_frequency.withColumn("Title", split(col("Name"), ", ")(1))
  .withColumn("Title", split(col("Title"), "\\.")(0))

// Add a new column 'Is_Married' and set it to 1 for 'Title' equal to 'Mrs'
val df_with_is_married: DataFrame = df_with_title.withColumn("Is_Married", when(col("Title") === "Mrs", 1).otherwise(0))

// Now df_with_is_married contains the 'Title' and 'Is_Married' columns

// COMMAND ----------

df_with_is_married.show()

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.util.regex.Pattern
import java.util.regex.Matcher

// Assuming you have a DataFrame df_all

// Initialize a SparkSession
val spark = SparkSession.builder()
  .appName("DataFrame Transformation")
  .getOrCreate()

// Define a user-defined function (UDF) to extract surnames
val extractSurname = udf((name: String) => {
  val nameNoBracket = if (name.contains("(")) name.split("\\(")(0) else name
  val family = nameNoBracket.split(",")(0).replaceAll("[^a-zA-Z]", "").trim
  family
})

// Add a new column 'Family' by applying the UDF to the 'Name' column
val df_with_family: DataFrame = df_with_is_married.withColumn("Family", extractSurname(col("Name")))

// Now df_train and df_test are the training and testing DataFrames, and df_with_family contains the 'Family' column

// COMMAND ----------

df_with_family.show()

// COMMAND ----------

//Indexing categorical features
val catFeatColNames = Seq("Pclass", "Sex", "Embarked")
val stringIndexers = catFeatColNames.map { colName =>
  new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "Indexed")
    .fit(trainingData)
}

//Indexing target feature
val labelIndexer = new StringIndexer()
.setInputCol("Survived")
.setOutputCol("SurvivedIndexed")
.fit(trainingData)

//Assembling features into one vector
val numFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
val idxdCatFeatColName = catFeatColNames.map(_ + "Indexed")
val allIdxdFeatColNames = numFeatColNames ++ idxdCatFeatColName
val assembler = new VectorAssembler()
  .setInputCols(Array(allIdxdFeatColNames: _*))
  .setOutputCol("Features")

// COMMAND ----------

//Randomforest classifier
val randomforest = new RandomForestClassifier()
  .setLabelCol("SurvivedIndexed")
  .setFeaturesCol("Features")

//Retrieving original labels
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labelsArray.flatten)

//Creating pipeline
val pipeline = new Pipeline().setStages(
  (stringIndexers :+ labelIndexer :+ assembler :+ randomforest :+ labelConverter).toArray)

// COMMAND ----------

//Selecting best model
val paramGrid = new ParamGridBuilder()
  .addGrid(randomforest.maxBins, Array(25, 28, 31))
  .addGrid(randomforest.maxDepth, Array(4, 6, 8))
  .addGrid(randomforest.impurity, Array("entropy", "gini"))
  .build()

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("SurvivedIndexed")
  .setMetricName("areaUnderPR")

//Cross validator with 10 fold
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)


// COMMAND ----------

//Fitting model using cross validation
val crossValidatorModel = cv.fit(trainingData)

//predictions on validation data
val predictions = crossValidatorModel.transform(validationData)

//Accuracy
val accuracy = evaluator.evaluate(predictions)
println("Test Error DT= " + (1.0 - accuracy))

// COMMAND ----------

////Implementing Gradient boosted tree
val gbt = new GBTClassifier()
  .setLabelCol("SurvivedIndexed")
  .setFeaturesCol("Features")
  .setMaxIter(10)

//Creating pipeline
val pipeline = new Pipeline().setStages(
  (stringIndexers :+ labelIndexer :+ assembler :+ gbt :+ labelConverter).toArray)

val model = pipeline.fit(trainingData)

val predictions = model.transform(validationData)

val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("SurvivedIndexed")
  .setMetricName("areaUnderPR")

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------


