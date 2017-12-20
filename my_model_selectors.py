import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(
                n_components=num_states,
                covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    Bayesian information criteria is defined as BIC = -2 * logL + p * logN, in which
        1. L is the likelihood of the fitted model
        2. p is the number of parameters
        3. N is the number of data points

    Reference: http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    """
    def get_num_params(n_components, d_data_points):
        """ calculate the number of components of GaussianHMM with
                1. diagonal covariance matrix for emission probability
                2. number of hidden states equals to n_components
        """
        # for transition matrix, each row should be summed up to 1:
        num_tran = n_components * (n_components - 1)

        # for emission probability, each dimension of observation has 2 params for each hidden state:
        num_emit = n_components * (2 * d_data_points)

        return num_tran + num_emit

    def get_bic(logL, p, logN):
        """ calculate BIC score

        """
        return -2 * logL + p * logN

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # config warning level:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # number of data points:
        N, D = self.X.shape
        logN = np.log(N)

        # performance:
        performance = []

        for n_components in range(
            self.min_n_components,
            self.max_n_components + 1
        ):
            # number of parameters:
            p = SelectorBIC.get_num_params(n_components, D)

            try:
                # build model on training set:
                model = GaussianHMM(
                    n_components = n_components,
                    covariance_type="diag",
                    n_iter=1000,
                    random_state=self.random_state,
                    verbose=False
                ).fit(
                    self.X,
                    self.lengths
                )
                # evaluate on test set:
                logL = model.score(
                    self.X,
                    self.lengths
                )
                # update performance list:
                performance.append(
                    (n_components, SelectorBIC.get_bic(logL, p, logN))
                )
            except Exception:
                continue

        # take max by average likelihood:
        if not performance:
            best_num_components = self.n_constant
        else:
            best_num_components = min(
                performance,
                key = lambda t: t[1]
            )[0]

        # retrain using whole dataset:
        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def get_other_logL(self, model):
        """ get goodness of fitness on other words
        """
        other_logL = []

        for other_word in self.words.keys():
            if other_word != self.this_word:
                # parse dataset for other word:
                X, lengths = self.hwords[other_word]

                # evaluate:
                try:
                    logL = model.score(
                        X, lengths
                    )
                except Exception:
                    continue

                other_logL.append(
                    logL
                )

        return np.mean(other_logL)

    def select(self):
        # config warning level:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # performance:
        performance = []

        for n_components in range(
            self.min_n_components,
            self.max_n_components + 1
        ):
            try:
                # build model on training set:
                model = GaussianHMM(
                    n_components = n_components,
                    covariance_type="diag",
                    n_iter=1000,
                    random_state=self.random_state,
                    verbose=False
                ).fit(
                    self.X,
                    self.lengths
                )
                # evaluate on this word:
                this_logL = model.score(
                    self.X,
                    self.lengths
                )

                # evaluate on other words:
                other_logL = self.get_other_logL(
                    model
                )
                # update performance list:
                performance.append(
                    (n_components, this_logL - other_logL)
                )
            except Exception:
                continue

        # take max by average likelihood:
        best_num_components = self.n_constant
        if performance:
            best_num_components = min(
                performance,
                key = lambda t: -t[1]
            )[0]

        # retrain using whole dataset:
        return self.base_model(best_num_components)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # initialize result:
        performance = {
            n_components: []
            for n_components in range(
                self.min_n_components,
                self.max_n_components + 1
            )
        }

        N = len(self.sequences)

        if N < 2:
            # for each hidden state number proposal:
            for n_components in range(
                self.min_n_components,
                self.max_n_components + 1
            ):
                try:
                    # build model on training set:
                    model = GaussianHMM(
                        n_components = n_components,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=self.random_state,
                        verbose=False
                    ).fit(
                        self.X,
                        self.lengths
                    )
                    # evaluate on test set:
                    performance[n_components].append(
                        model.score(
                            self.X,
                            self.lengths
                        )
                    )
                except Exception:
                    pass
        else:
            # k-fold generator:
            k_fold = KFold(
                n_splits = min(len(self.sequences),5),
                random_state=self.random_state
            )
            for train_index, test_index in k_fold.split(self.sequences):
                # training set:
                X_train, lengths_train = combine_sequences(train_index, self.sequences)
                # testing set:
                X_test, lengths_test = combine_sequences(test_index, self.sequences)

                # for each hidden state number proposal:
                for n_components in range(
                    self.min_n_components,
                    self.max_n_components + 1
                ):
                    try:
                        # build model on training set:
                        model = GaussianHMM(
                            n_components = n_components,
                            covariance_type="diag",
                            n_iter=1000,
                            random_state=self.random_state,
                            verbose=False
                        ).fit(
                            X_train,
                            lengths_train
                        )
                        # evaluate on test set:
                        performance[n_components].append(
                            model.score(
                                X_test,
                                lengths_test
                            )
                        )
                    except Exception:
                        pass

        # get average log likelihood:
        performance = [
            (n_components, np.mean(logLs)) for n_components, logLs in performance.items() if len(logLs) != 0
        ]

        # take num. hidden states of max average likelihood:
        best_num_components = self.n_constant
        if performance:
            # take max by average likelihood:
            best_num_components = min(
                performance,
                key = lambda t: -t[1]
            )[0]

        # retrain using whole dataset:
        return self.base_model(best_num_components)
