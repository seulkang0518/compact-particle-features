"""Schedule/restart invariants for matched one/four-rank training."""
import unittest

from train import episode_schedule, learning_rate


class ScheduleTests(unittest.TestCase):
    def test_same_global_episode_order_after_partition(self):
        whole = episode_schedule(50, 4, 3, 42)
        self.assertEqual(whole, sum([whole[i:i+1] for i in range(4)], []))
        self.assertEqual(whole, episode_schedule(50, 4, 3, 42))
        self.assertNotEqual(whole, episode_schedule(50, 4, 4, 42))

    def test_warmup_and_cosine_endpoints(self):
        cfg = {"warmup_steps": 2, "learning_rate": .001}
        self.assertAlmostEqual(learning_rate(0, 10, cfg), .0005)
        self.assertAlmostEqual(learning_rate(1, 10, cfg), .001)
        self.assertAlmostEqual(learning_rate(2, 10, cfg), .001)
        self.assertAlmostEqual(learning_rate(9, 10, cfg), 0)


if __name__ == "__main__":
    unittest.main()
