"""Run with: python -m unittest discover -s tests -v"""
import ast
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.recommenders.mf_fair import FairMF


class NewUserInferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = FairMF(batch_size=100, num_factors=2, max_epochs=350,
                            learning_rate=0.05, l2=0.1, patience=30,
                            min_delta=1e-9, seed=42)
        self.model.device = torch.device('cpu')
        self.model._init_model(csr_matrix((6, 4)))
        with torch.no_grad():
            # Both provider groups contain the same factors, so non-parity is
            # exactly zero. The unmasked ridge optimum is available analytically.
            self.model.model_.item_embedding.weight.copy_(torch.tensor(
                [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
        self.X = csr_matrix([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.groups = torch.tensor([[False, False, True, True]] * 2)

    def test_unmasked_objective_matches_closed_form_optimum(self):
        # Let Adam converge fully for the analytic comparison, independently
        # of the production early-stopping tolerance.
        self.model.patience = self.model.max_epochs
        scores = self.model.predict_new_users(self.X, self.groups).toarray()
        V = self.model.model_.item_embedding.weight.detach().numpy()
        # d/dU [||X-UV^T||^2 / (n*m) + l2*||U||^2/2] = 0.
        ridge = self.model.l2_lambda * np.prod(self.X.shape) / 2
        U = np.linalg.solve(V.T @ V + ridge * np.eye(2),
                            (self.X @ V).T).T
        np.testing.assert_allclose(scores, U @ V.T, atol=2e-4)
        # Items 2 and 3 were not observed. They nevertheless enter the loss.
        self.assertGreater(scores[0, 2], scores[0, 3])
        self.assertGreater(scores[1, 3], scores[1, 2])

    def test_histories_and_row_order_control_predictions(self):
        first = self.model.predict_new_users(self.X, self.groups).toarray()
        changed = self.model.predict_new_users(self.X[::-1], self.groups).toarray()
        self.assertFalse(np.allclose(first, changed))
        np.testing.assert_allclose(changed, first[::-1], atol=1e-6)
        repeated = self.model.predict_new_users(self.X, self.groups).toarray()
        np.testing.assert_array_equal(first, repeated)

    def test_fitted_model_training_state_and_rng_are_unchanged(self):
        self.model.epochs = 17
        self.model.best_loss = 0.123
        self.model.patience_counter = 2
        before = copy.deepcopy(self.model.model_.state_dict())
        optimizer = copy.deepcopy(self.model.optimizer.state_dict())
        old_predictions = self.model.predict(csr_matrix(np.ones((6, 4)))).toarray()
        self.model.predict_new_users(self.X, self.groups)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.model.model_.state_dict()[name]))
        self.assertEqual(optimizer, self.model.optimizer.state_dict())
        self.assertEqual(self.model.epochs, 17)
        self.assertEqual(self.model.steps, 0)
        self.assertEqual(self.model.best_loss, 0.123)
        self.assertEqual(self.model.patience_counter, 2)
        self.assertTrue(all(p.grad is None for p in self.model.model_.parameters()))
        # predict() uses a DataLoader and advances the RNG; capture immediately
        # before inference to isolate the new path.
        np.testing.assert_array_equal(
            old_predictions, self.model.predict(csr_matrix(np.ones((6, 4)))).toarray())
        rng = torch.random.get_rng_state().clone()
        self.model.predict_new_users(self.X, self.groups)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_empty_histories_and_input_validation(self):
        X = csr_matrix([[0., 0., 0., 0.], [1., 0., 0., 0.]])
        scores = self.model.predict_new_users(X, self.groups).toarray()
        np.testing.assert_array_equal(scores[0], np.zeros(4))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(self.model.predict_new_users(
            csr_matrix((0, 4)), torch.empty((0, 4), dtype=torch.bool)).shape, (0, 4))
        with self.assertRaises(ValueError):
            self.model.predict_new_users(csr_matrix((2, 5)), self.groups)
        with self.assertRaises(ValueError):
            self.model.predict_new_users(self.X, self.groups[:1])

    def test_training_loss_still_includes_zeros_and_fairness(self):
        true = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        pred = torch.tensor([[0.2, 0.8, 0.4, 0.6], [0.1, 0.9, 0.3, 0.5]])
        U = self.model.model_.user_embedding.weight
        V = self.model.model_.item_embedding.weight
        expected = ((pred - true)**2).mean() + 0.05 * (U.square().sum() + V.square().sum())
        expected += torch.nn.functional.smooth_l1_loss(
            pred[~self.groups].mean(), pred[self.groups].mean())
        torch.testing.assert_close(self.model._compute_loss(true, pred, self.groups), expected)

    def test_new_user_predictions_do_not_depend_on_old_user_factors(self):
        expected = self.model.predict_new_users(self.X, self.groups).toarray()
        with torch.no_grad():
            self.model.model_.user_embedding.weight.fill_(1000.)
        actual = self.model.predict_new_users(self.X, self.groups).toarray()
        np.testing.assert_array_equal(actual, expected)


class NotebookWiringTests(unittest.TestCase):
    def test_all_provider_notebooks_use_input_only_inference(self):
        root = Path(__file__).resolve().parents[1]
        for name in ['Coco_all.ipynb', 'Goodreads.ipynb']:
            with self.subTest(notebook=name):
                notebook = json.loads((root / 'notebooks' / name).read_text())
                calls = []
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    tree = ast.parse(''.join(cell['source']))
                    calls.extend(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                                 and isinstance(n.func, ast.Attribute)
                                 and n.func.attr == 'predict_new_users')
                self.assertEqual(len(calls), 3)  # Two objectives and final test.
                self.assertEqual([n.args[0].id for n in calls].count(
                    'filtered_validation_data_in'), 2)
                self.assertEqual([n.args[0].id for n in calls].count(
                    'filtered_test_data_in'), 1)
                for call in calls:
                    self.assertEqual(ast.unparse(call.args[1]), 'sst_field[valid_rows]')


if __name__ == '__main__':
    unittest.main()
