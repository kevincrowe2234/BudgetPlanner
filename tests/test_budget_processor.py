import unittest
import pandas as pd
from datetime import datetime, timedelta
from logic.budget_processor import prepare_dataframes_for_projection, generate_timeline

class TestBudgetProcessor(unittest.TestCase):
    def test_prepare_dataframes_empty(self):
        """Test that prepare_dataframes_for_projection handles empty inputs"""
        income_df = pd.DataFrame()
        expense_df = pd.DataFrame()
        
        income_clean, expense_clean = prepare_dataframes_for_projection(income_df, expense_df)
        
        self.assertTrue(income_clean.empty)
        self.assertTrue(expense_clean.empty)
    
    def test_prepare_dataframes_with_data(self):
        """Test that prepare_dataframes_for_projection properly handles dataframes with data"""
        income_df = pd.DataFrame({
            'Description': ['Salary'],
            'Amount': [1000.0],
            'Date': [datetime.now()],
            'Recurring': [True],
            'Frequency': ['Monthly']
        })
        
        expense_df = pd.DataFrame({
            'Description': ['Rent'],
            'Amount': [500.0],
            'Date': [datetime.now()],
            'Recurring': [True],
            'Frequency': ['Monthly']
        })
        
        income_clean, expense_clean = prepare_dataframes_for_projection(income_df, expense_df)
        
        self.assertEqual(len(income_clean), 1)
        self.assertEqual(len(expense_clean), 1)

if __name__ == '__main__':
    unittest.main()