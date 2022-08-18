from sklearn.datasets import make_blobs


def create_random_sample(num_samples, num_features):
    return make_blobs(n_samples=num_samples, n_features=num_features)


def log(log_str, flag):
    if flag:
        print(log_str)
