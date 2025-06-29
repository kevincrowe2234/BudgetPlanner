import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from logic.projection_engine import calculate_financial_metrics

class TestProjectionEngine(unittest.TestCase):
    def test_calculate_financial_metrics(self):
        """Test that calculate_financial_metrics returns correct values"""
        # Create sample monthly data
        monthly_df = pd.DataFrame({
            'Income': [1000.0, 1000.0, 1000.0],
            'Expense': [800.0, 900.0, 1100.0],
            'Net': [200.0, 100.0, -100.0],
            'Balance': [1200.0, 1300.0, 1200.0]
        })
        
        metrics = calculate_financial_metrics(monthly_df)
        
        self.assertAlmostEqual(metrics['average_monthly_income'], 1000.0)
        self.assertAlmostEqual(metrics['average_monthly_expenses'], 933.33, places=2)
        self.assertAlmostEqual(metrics['average_monthly_savings'], 66.67, places=2)
        self.assertAlmostEqual(metrics['savings_rate'], 6.67, places=2)
        self.assertEqual(metrics['months_positive'], 2)
        self.assertEqual(metrics['months_negative'], 1)

if __name__ == '__main__':
    unittest.main()