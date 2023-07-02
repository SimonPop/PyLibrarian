from surprise import accuracy, Dataset, SVD
from surprise.model_selection import train_test_split

from dataset import PackageDataset

# Load the movielens-100k dataset (download it if needed),
dataset = PackageDataset(project="the-sandbox-386618")
df = dataset.load_data()
df = dataset.to_surprise(df)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(df, test_size=0.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)