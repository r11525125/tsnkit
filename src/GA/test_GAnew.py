import unittest
from GAnew import Scheduler, task_var

class GANewTest(unittest.TestCase):
    def test_scheduler(self):
        # Test case 1: Single task with one path
        task_var[0]['paths'] = ['(1, 2)']
        self.assertTrue(Scheduler(0))
        self.assertEqual(task_var[0]['arr'], 0)
        self.assertEqual(task_var[0]['IT'], 0)
        self.assertEqual(task_var[0]['r'], '(1, 2)')

        # Test case 2: Single task with multiple paths
        task_var[1]['paths'] = ['(1, 2)', '(2, 1)']
        self.assertTrue(Scheduler(1))
        self.assertEqual(task_var[1]['arr'], 0)
        self.assertEqual(task_var[1]['IT'], 0)
        self.assertIn(task_var[1]['r'], ['(1, 2)', '(2, 1)'])

        # Test case 3: Multiple tasks with multiple paths
        task_var[0]['paths'] = ['(1, 2)']
        task_var[1]['paths'] = ['(2, 1)']
        self.assertTrue(Scheduler(0))
        self.assertTrue(Scheduler(1))
        self.assertEqual(task_var[0]['arr'], 0)
        self.assertEqual(task_var[0]['IT'], 0)
        self.assertEqual(task_var[0]['r'], '(1, 2)')
        self.assertEqual(task_var[1]['arr'], 0)
        self.assertEqual(task_var[1]['IT'], 0)
        self.assertEqual(task_var[1]['r'], '(2, 1)')

if __name__ == '__main__':
    unittest.main()