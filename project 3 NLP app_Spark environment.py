# Read in data from S3 Buckets
from pyspark import SparkFiles

from pyspark import sql, SparkConf, SparkContext

url = 'https://bucket1-jp.s3.us-east-2.amazonaws.com/kpr437.csv'
spark.sparkContext.addFile(url)
df = spark.read.csv(SparkFiles.get("kpr437.csv"), sep=",", encoding ='utf-8', header=True)

from pyspark.sql.functions import length
# Create a length column to be used as a future feature 
data_df = df.withColumn('length', length(df['Customer Reviews']))

stop_list = ['Santa Ana','DTSA', 'Downtown', 'Mediterranean', 'Lebanese', 'Arabic', 'Middle Eastern','Persian', 'Iranian', 'street',
             'Rok', '4th Street','parking', 'San Diego', 'Irvine', 'Grubhub', 'Doordash', 'Uber Eats', 'Kebab Place', 'Postmates', 
             'Yelp']

stop_list_two = ['OC', 'Orange County', 'jury duty', 'Starbucks', 'starbucks']

#Feature Transformation
from pyspark.ml.feature import Tokenizer, NGram, StopWordsRemover, HashingTF, IDF, StringIndexer
# Create all the features to the data set
pos_neg_to_num = StringIndexer(inputCol='Class',outputCol='label')
tokenizer = Tokenizer(inputCol='Customer Reviews', outputCol='token_review')
ngramzero = NGram(n=3, inputCol='token_review', outputCol='ngrammedzero')
stopremove = StopWordsRemover(inputCol='ngrammedzero',outputCol='stop_tokens')
stopremoveTWO = StopWordsRemover(inputCol='stop_tokens', outputCol='more_stop_tokens', stopWords = stop_list)
stopremoveTHREE = StopWordsRemover(inputCol='more_stop_tokens', outputCol='one_more_stop_tokens', stopWords = stop_list_two)
ngram = NGram(n=3, inputCol='one_more_stop_tokens', outputCol='ngrammed')
hashingTF = HashingTF(inputCol='ngrammed', outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

#Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

from pyspark.ml import Pipeline
data_prep_pipeline = Pipeline(stages=[pos_neg_to_num, tokenizer, ngramzero, stopremove, stopremoveTWO, stopremoveTHREE, ngram, hashingTF, idf, clean_up])

# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(data_df)
cleaned = cleaner.transform(data_df)

from pyspark.ml.classification import NaiveBayes
# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])
# Create a Naive Bayes model and fit training data
nb = NaiveBayes()
predictor = nb.fit(training)

# Tranform the model with the testing data
test_results = predictor.transform(testing)

# Use the Class Evaluator for a cleaner description
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)