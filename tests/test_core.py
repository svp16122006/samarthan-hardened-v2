import unittest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import RISK_ORDER, action_for

class CoreTests(unittest.TestCase):
    def test_risk_order(self):
        self.assertLess(RISK_ORDER['stable'], RISK_ORDER['critical'])
    def test_action_mapping(self):
        self.assertIn('human', action_for('critical').lower())
        self.assertIn('Routine', action_for('stable'))

if __name__ == '__main__': unittest.main()
