import unittest
from pathlib import Path
from unittest.mock import patch

from trainer import ExperimentTracker


class ExperimentTrackerValLossTest(unittest.TestCase):
    def test_log_epoch_preserves_missing_val_loss_as_none(self):
        tracker = ExperimentTracker("dummy")

        tracker.log_epoch(
            epoch=1,
            train_loss=1.23456,
            val_loss=None,
            val_metrics={"HR@10": 0.5, "NDCG@10": 0.25},
        )

        self.assertIsNone(tracker.epochs[0]["val_loss"])
        self.assertEqual(tracker.epochs[0]["train_loss"], 1.23456)
        self.assertEqual(tracker.epochs[0]["HR@10"], 0.5)

    def test_plot_losses_skips_val_curve_when_val_loss_missing(self):
        tracker = ExperimentTracker("dummy")
        tracker.log_epoch(
            epoch=1,
            train_loss=1.0,
            val_loss=None,
            val_metrics={"HR@10": 0.5},
        )

        plotted_labels = []

        def capture_plot(*args, **kwargs):
            label = kwargs.get("label")
            if label:
                plotted_labels.append(label)

        with patch("trainer.plt.plot", side_effect=capture_plot), \
             patch("trainer.plt.savefig"), \
             patch("trainer.plt.close"):
            tracker.plot_losses(Path("dummy.png"))

        self.assertIn("Train Loss", plotted_labels)
        self.assertNotIn("Val Loss", plotted_labels)


if __name__ == "__main__":
    unittest.main()
