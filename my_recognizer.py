import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    # config warning level:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # initialize output:
    probabilities = []
    guesses = []

    # for each test word:
    for word_id in range(test_set.num_items):
        # parse dataset:
        X, lengths = test_set.get_item_Xlengths(word_id)

        word_logL = {}
        best_logL = float('-inf'); best_word = None
        for word, model in models.items():
            try:
                word_logL[word] = model.score(
                    X, lengths
                )
            except Exception:
                word_logL[word] = float('-inf')

            if word_logL[word] > best_logL:
                best_logL = word_logL[word]
                best_word = word

        probabilities.append(word_logL)
        guesses.append(best_word)

    return (probabilities, guesses)
