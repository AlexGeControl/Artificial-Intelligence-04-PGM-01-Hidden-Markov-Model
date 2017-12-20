# Sign Language Recognition using Gaussian HMM

## PART 0: Raw Data

The data in the `asl_recognizer/data/` directory was derived from
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php).
The handpositions (`hand_condensed.csv`) are pulled directly from
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv`
file.

## PART 1: Feature Engineering

### What custom feature set was chosen and why?

Here three groups of features are derived, namely:

1. normalized ground coordinates
2. normalized polar coordinates
3. delta differences of both normalized ground coordinates & polar coordinates

**Normalized ground coordinates** are derived by normalizing ground features with mean and standard deviation statistics. Normalized ground features could better decrease the variance introduced by varing speaker physical attributes such as height, weight, arm length, etc. The Python script for this group of features is as follows:

```python
  features_norm_grnd = [
      'norm-grnd-rx',
      'norm-grnd-ry',
      'norm-grnd-lx',
      'norm-grnd-ly'
  ]

  # extract speaker statistics:
  # a. means:
  for group in ['grnd-ry', 'grnd-rx', 'grnd-ly', 'grnd-lx']:
      asl.df[
          "{}-mean".format(group)
      ] = asl.df['speaker'].map(df_means[group])

  # b. stds:
  for group in ['grnd-ry', 'grnd-rx', 'grnd-ly', 'grnd-lx']:
      asl.df[
          "{}-std".format(group)
      ] = asl.df['speaker'].map(df_std[group])

  # generate:
  for group in ['ry', 'rx', 'ly', 'lx']:
      asl.df["norm-grnd-{}".format(group)] = (
          asl.df["grnd-{}".format(group)] - asl.df["grnd-{}-mean".format(group)]
      ) / asl.df["grnd-{}-std".format(group)]

  # verification -- no division by zero:
  print(
      "[Total number of NaN normalized feature]: {}".format(
          sum([asl.df["norm-grnd-{}".format(group)].isnull().any() for group in ['ry', 'rx', 'ly', 'lx']])
      )
  )
```

**Normalized polar coordinates** are derived by converting normalized ground Cartesian coordinates to polar coordinates. Polar coordinates are good at describing circular motion of speaker hands. The Python snippet for this group of features are below:

```python
  # normalized polar feature set:
  features_norm_polar = [
      'norm-polar-rr',
      'norm-polar-rtheta',
      'norm-polar-lr',
      'norm-polar-ltheta'
  ]

  # generate:
  for group in ['l', 'r']:
      asl.df["norm-polar-{}r".format(group)] = np.linalg.norm(
          asl.df[
              ["norm-grnd-{}x".format(group), "norm-grnd-{}y".format(group)]
          ].values,
          axis = 1
      )
      asl.df["norm-polar-{}theta".format(group)] = np.arctan2(
          asl.df["norm-grnd-{}x".format(group)].values,
          asl.df["norm-grnd-{}y".format(group)].values
      )
```

**Delta differences of both normalized ground coordinates & polar coordinates** are derived by subtracting previous observation from current one on both normalized ground & polar coordinates. This group of features are good at capturing motion speed related features of speaker hands. The Python snippet for this group of features are as follows:

```python
  # normalized diff feature set:
  features_norm_delta = [
      'norm-delta-rx',
      'norm-delta-ry',
      'norm-delta-lx',
      'norm-delta-ly',
      'norm-delta-rr',
      'norm-delta-rtheta',
      'norm-delta-lr',
      'norm-delta-ltheta'
  ]

  # generate:
  for group in ['ry', 'rx', 'ly', 'lx']:
      asl.df["norm-delta-{}".format(group)] = asl.df["norm-grnd-{}".format(group)].diff().fillna(0)
  for group in ['rr', 'rtheta', 'lr', 'ltheta']:
      asl.df["norm-delta-{}".format(group)] = asl.df["norm-polar-{}".format(group)].diff().fillna(0)
```

## PART 2: Model Selection

All the three model selectors are implemented inside <a href="my_model_selectors.py">my_model_selectors.py</a>

Here special cares should be taken to handle the exceptions caused by failed model training & scoring.

### Cross-Validation Selector

```python
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
```

### Bayesian Information Criterion(BIC) Selector

```python
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
```

### Discriminative Information Criterion(DIC) Selector

```python
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
```

### Model Selector Comparison

The respective pros & cons of the above three model selectors are summarized as follows:

**CV** selector has the most accurate estimation of likelihood, the generative power of the fitted model. However, it also has the longest evaluation time due to its use of cross validation, tends to overfitting since no model complexity penalty is used and tells nothing about model discriminative power.

**BIC** selector is the quickest to evaluate and leads to simpler model. However, its likelihood estimation might not be accurate, its model complexity regularization might not be effective in certain situations and it also tells nothing about model discriminative power.

**DIC** selector prefer discriminative model thus good for classification task. However, its likelihood estimation might not be accurate and it might not scale well when there are too many target classes

## PART 3: Recognizer

### Recognizer Performance

WER for the 6 evaluated feature set & model selector combinations are as follows:

| WER | All Pre-Defined Feature Set | Custom Feature Set |
|:---:|:---------------------------:|:------------------:|
| DIC |            0.5112           |       0.4944       |
| BIC |            0.4944           |       0.4663       |
|  CV |            0.5337           |       0.4888       |

### Performance Analysis

**For the two set of features**, pre-defined feature set performs better than my self-defined feature set.

**For the three model selectors**, BIC performs the best, after it comes the DIC and CV has the worst performance.

So **model trained using pre-defined feature set with BIC model selector is the best combination**. I think the simpler model attained through BIC helps the model to generalize better on testing dataset.

**For further improvement**, I think n-gram model and deep learning methods such as LSTM could be used to attain better model performance.
