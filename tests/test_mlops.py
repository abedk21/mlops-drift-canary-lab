import json
import math
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from mlops_lab.core import (
    Model, Policy, canary_gate, clamp, drift, experiment, fit, psi, roc_auc,
    samples, score, sigmoid,
)
from mlops_lab.report import render


class MLOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = samples(450, 13)
        cls.monitor = samples(200, 14)
        cls.model = fit(cls.training[:400])
        cls.stats = score(cls.model, cls.monitor)

    def test_reproducible_generator(self):
        self.assertEqual(samples(20,42),samples(20,42))

    def test_different_seed_changes_traffic(self):
        self.assertNotEqual(samples(20,41),samples(20,42))

    def test_generated_features_normalized(self):
        for row in samples(100,1,shifted=True):
            self.assertTrue(all(0 <= v <= 1 for v in row['features'].values()))

    def test_generator_validation(self):
        for n in (0,-1,10001):
            with self.assertRaises(ValueError): samples(n,2)

    def test_model_has_three_coefficients(self):
        self.assertEqual(len(self.model.weights),3)
        self.assertEqual(self.model.trained_examples,400)

    def test_model_training_is_deterministic(self):
        self.assertEqual(self.model,fit(self.training[:400]))

    def test_trained_model_beats_constant_guess(self):
        bad = Model((0,0,0),-8,400)
        self.assertGreater(self.stats['f1'], score(bad,self.monitor)['f1'])

    def test_model_probabilities_are_valid(self):
        self.assertTrue(all(0 < self.model.predict(x['features']) < 1 for x in self.monitor))

    def test_model_rejects_bad_features(self):
        for val in (float('nan'),-1,3):
            with self.assertRaises(ValueError):
                self.model.predict({**self.monitor[0]['features'],'signal_strength':val})

    def test_sigmoid_extreme_stable(self):
        self.assertAlmostEqual(sigmoid(900),1)
        self.assertAlmostEqual(sigmoid(-900),0)

    def test_confusion_totals(self):
        self.assertEqual(sum(self.stats['confusion'].values()),len(self.monitor))

    def test_auc_limits(self):
        self.assertTrue(0<=self.stats['auc']<=1)
        self.assertEqual(roc_auc([0,1],[0.0,1.0]),1)
        self.assertEqual(roc_auc([1,0],[0.0,1.0]),0)

    def test_auc_with_tied_predictions(self):
        self.assertAlmostEqual(roc_auc([1,0],[.5,.5]),.5)

    def test_auc_unavailable_single_class(self):
        self.assertIsNone(roc_auc([1,1],[.5,.7]))

    def test_score_refuses_empty_or_bad_threshold(self):
        with self.assertRaises(ValueError): score(self.model,[])
        with self.assertRaises(ValueError): score(self.model,self.monitor,threshold=1.0)

    def test_slice_recall_values_in_range(self):
        for x in self.stats['segments'].values():
            self.assertTrue(x['recall'] is None or 0<=x['recall']<=1)

    def test_psi_no_drift_is_zero(self):
        a=[0.05,0.2,0.4,0.6,0.8,0.99]
        self.assertAlmostEqual(psi(a,a),0)

    def test_psi_detects_obvious_shift(self):
        self.assertGreater(psi([.1]*30,[.9]*30),1)

    def test_psi_rejects_invalid(self):
        with self.assertRaises(ValueError): psi([], [.1])
        with self.assertRaises(ValueError): psi([float('inf')],[.1])
        with self.assertRaises(ValueError): psi([.1], [.2],bins=1)

    def test_shifted_data_has_more_drift(self):
        ref=self.training
        normal=drift(ref,self.monitor)
        shifted=drift(ref,samples(200,14,shifted=True))
        self.assertTrue(all(shifted[k] > normal[k] for k in normal))

    def test_healthy_candidate_allowed(self):
        result=experiment('healthy')
        self.assertEqual(result['canary_gate']['decision'],'allow_candidate')

    def test_shifted_traffic_blocks_candidate(self):
        result=experiment('shifted')
        self.assertEqual(result['canary_gate']['decision'],'hold_candidate')
        self.assertIn('psi_signal_strength',[x['check'] for x in result['canary_gate']['checks'] if not x['passed']])

    def test_bad_model_blocks_candidate(self):
        result=experiment('degraded')
        self.assertEqual(result['canary_gate']['decision'],'hold_candidate')
        self.assertIn('f1_regression',[x['check'] for x in result['canary_gate']['checks'] if not x['passed']])

    def test_gate_never_deploys(self):
        result=experiment('healthy')
        self.assertFalse(result['canary_gate']['actual_deployment_performed'])

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(ValueError):experiment('not-real')

    def test_too_small_segment_fails_closed(self):
        metrics=self.stats.copy()
        metrics['segments']={'tiny':{'positives':0,'recall':None,'brier':0}}
        gate=canary_gate(metrics,metrics,{'signal_strength':0})
        self.assertEqual(gate['decision'],'hold_candidate')

    def test_f1_regression_policy_can_block(self):
        candidate=self.stats.copy()
        candidate['f1']=0.1
        gate=canary_gate(self.stats,candidate,{'signal_strength':0})
        self.assertEqual(gate['decision'],'hold_candidate')

    def test_serializable_model_artifact(self):
        payload=json.dumps(self.model.as_dict())
        parsed=json.loads(payload)
        self.assertEqual(parsed['training_method'],'batch_gradient_descent_logistic_regression')
        self.assertEqual(len(parsed['features']),3)

    def test_html_report_escape_and_sections(self):
        result=experiment('healthy')
        result['scenario']='<img src=x onerror=alert(1)>'
        html=render(result)
        self.assertNotIn('<img src=x',html)
        self.assertIn('&lt;img src=',html)
        self.assertIn('Feature drift',html)
        self.assertIn('Canary',html)

    def test_l2_and_training_params_validation(self):
        with self.assertRaises(ValueError):fit(self.training[:10])
        with self.assertRaises(ValueError):fit(self.training[:50],epochs=0)
        with self.assertRaises(ValueError):fit(self.training[:50],lr=-1)

    def test_model_training_rejects_invalid_binary_label(self):
        bad=[dict(x) for x in self.training[:30]]
        bad[1]={**bad[1],'label':2}
        with self.assertRaises(ValueError):fit(bad)


if __name__ == '__main__':
    unittest.main()
