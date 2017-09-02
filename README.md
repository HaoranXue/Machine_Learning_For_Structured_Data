# Machine Learning for Structured Data- MLSD

Author: Haoran Xue

This Python library is developed for my master thesis supervised by Dr. Franz Kiraly at University College London.

In this python library,  we provide a data container and Multiply transformers for Structured data.

The data container is for the storage of Structured data such as: Series data, Bag of features, List of image or text. By using the data container we can store the different types of structured data in one Data frame and doing features extractions automatically for common machine learning models such as supervised learning or unsupervised learning in Sci-Kit learn package.

There are three parts in my data container package, The first one is SData which is the container for each structured feature and the columns for SDataFrame. SData can also store the transformer for this structured feature. The second part is SDataFrame class which is the data frame to contain all features.  The last part is the transformers, which can be stored in the SData class, different types of SData have different default transformers, however, it can be replaced by any transformers in the package.
