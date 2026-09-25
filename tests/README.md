# FairMF strong-generalization regression checks

Run from the repository root in the experiment environment:

```sh
python -m unittest discover -s tests -v
```

`FairMF.predict(X)` is for known users with their original training row IDs.
`FairMF.predict_new_users(X_in, sst_field)` infers fresh user factors from
validation/test input histories, keeping the training item factors fixed.
Item columns must retain their training order, and the sensitive-field rows
must be filtered alongside the interaction rows. Held-out target interactions
must never be passed to inference.

The new path uses the existing unmasked MSE, L2, and non-parity objective. It
uses the model's configured learning rate and stopping settings. User factors
start at zero; the fitted model and its training statistics remain unchanged.
The provider-fairness notebooks use this path in both tuning objectives
and final evaluation. They save prediction time (including user-factor
inference for FairMF) separately as `predict_time`; `fit_time` still measures
training only.

These tests check history dependence, row alignment, isolation from fitted
user factors, preservation of the fitted model and unmasked objective, and
agreement with an analytic ridge solution where non-parity is zero. They also
check that the provider notebooks pass input histories and aligned group
fields to inference.

Existing saved results and tuned hyperparameters predate this fix. Retune and
rerun FairMF for COCO and Goodreads before using corrected comparisons. These
regression checks do not regenerate experimental results.
