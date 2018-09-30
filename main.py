import numpy as np
from math import log, exp

# Assume ham and spam matrix is already built
ham = np.array(
    [
        [1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 1]
    ]
)

spam = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
    ]
)


# First step: Compute the priors
def get_priors(ham, spam):
    total = ham.shape[1] + spam.shape[1]
    return {
        'ham': ham.shape[1]/total,
        'spam': spam.shape[1]/total
    }


# Second step: Build Likelihood (might want to save this as well boys)
def get_likelihoods(ham, spam):
    ham_likelihood = []
    spam_likelihood = []
    ham_count = ham.shape[1]
    spam_count = spam.shape[1]
    for row in ham:
        # Likelihood of 1
        ham_likelihood.append(np.count_nonzero(row)/ham_count)

    for row in spam:
        # Likelihood of 1
        spam_likelihood.append(np.count_nonzero(row)/spam_count)

    return {
        'ham': ham_likelihood,
        'spam': spam_likelihood
    }


# Third step: Do the actual computation
def classify(document, ham, spam):
    priors = get_priors(ham, spam)
    likelihoods = get_likelihoods(ham, spam)
    ham_probability = 0
    spam_probability = 0
    for (index, value) in enumerate(document):
        ham_prob = likelihoods['ham'][index]
        spam_prob = likelihoods['spam'][index]
        if not value:
            ham_prob = 1 - ham_prob
            spam_prob = 1 - spam_prob
        ham_probability += log(ham_prob)
        spam_probability += log(spam_prob)

    ham_probability = exp(ham_probability) * priors['ham']
    spam_probability = exp(spam_probability) * priors['spam']
    return ham_probability / (ham_probability + spam_probability)


document_1 = [1, 0, 0, 1, 1, 1, 0, 1]
document_2 = [0, 1, 1, 0, 1, 0, 1, 0]
document_3 = [1, 0, 0, 1, 1, 1, 1, 1]
print(float(classify(document_3, ham, spam)))
