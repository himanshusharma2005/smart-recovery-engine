# Day 4 - Training the retry-success classifier

## What was built

`src/train_model.py` - trains and compares two classifiers on
`features.csv`: Logistic Regression (interpretable baseline) and Random
Forest (usually stronger on non-linear patterns). Picks the winner by
ROC-AUC, not accuracy alone, and saves it to `data/generated/model.pkl` for
Day 6's simulation engine to load directly.

## Why compare two models instead of just picking one

Random Forest is often assumed to be the "better" choice by default, but
that's not something to take on faith - it needs to actually be measured.
Here, Logistic Regression won (ROC-AUC 0.736 vs 0.720). This makes sense
given the data: the strongest signal (decline code / category) is
essentially categorical and close to linearly separable once one-hot
encoded, so a linear model captures it just as well as a more complex one,
with the added benefit of being easier to explain to a non-technical
person (an interviewer, a judge, a merchant).

## Why ROC-AUC instead of just accuracy

Accuracy alone (66% for both models) doesn't say anything about *how
confident* the model is or how well it separates the two classes across
different thresholds - a model that always predicts "success" would score
~50% accuracy here by coincidence, given the near-balanced target. ROC-AUC
of 0.736 means: given one random failed-retry and one random
successful-retry, the model correctly ranks the successful one higher
about 74% of the time. That is a real, learned pattern - notably better
than a coin flip (0.5) - even though it's far from perfect, which is
honest and expected given how much randomness we deliberately built into
the synthetic data generator (Day 2's +/-5% noise term).

## Bug avoided (not fixed - avoided, on purpose)

The first run threw a `ConvergenceWarning` for Logistic Regression. Root
cause: numeric columns like `subscription_amount` (values up to 1499) sit
on a wildly different scale than binary columns like `is_upi` (0 or 1),
which makes the solver struggle. Fixed by applying `StandardScaler` to the
numeric columns only - fit on the training set only, then applied to the
test set, to avoid data leakage (fitting a scaler on test data would let
information about the test set quietly influence training, which inflates
reported performance in a way that wouldn't hold up in the real world).

## Feature importance findings

The top features are `category_hard_decline`, `category_soft_decline`, and
`decline_CARD_EXPIRED` - this independently confirms Day 3's EDA finding
that decline category is the dominant signal for retry success. Seeing the
same pattern show up twice, once from manual analysis and once from a
trained model, is a good sanity check that neither step was a fluke.

See `docs/images/feature_importance.png`.

## Honest limitation to mention if asked

74% ROC-AUC is good, not exceptional - and that's expected here, since the
"ground truth" itself was generated with deliberate randomness (Day 2) to
avoid an unrealistically clean synthetic dataset. A real production model
trained on Razorpay's actual first-party transaction data would likely
perform better, since it would have access to signals this synthetic
dataset can't include (actual customer behavior history, real-time bank
status, etc.).

## What's next (Day 5)

Build the rule-based scheduler - encodes hard constraints the ML model
doesn't know about (card network retry caps, escalation thresholds) and
combines with the model's probability predictions to make an actual
retry/escalate decision per transaction.
