from unittest import TestCase, mock
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd

from recommender import Recommender
from tests.config import Config
from train_model import TrainModel


class TestTrainModel(TestCase):
    def setUp(self):
        self.config = Config()
        self.test_train_model = TrainModel(self.config)
        self.dummy_data_df = pd.DataFrame(
            np.array([
                ['val11', 'val12', 'val13'],
                ['val21', 'val22', 'val23'],
                ['val31', 'val32', 'val33']
            ]),
            columns=self.config.ALL_PRODUCT_ATTRS)

    @mock.patch.object(Recommender, 'fit')
    def test_train(self, mocked_fit):

        # mock calls.
        self.test_train_model.read_data = MagicMock(return_value=self.dummy_data_df)
        self.test_train_model.write_data = MagicMock(return_value=None)
        self.test_train_model.transform_data = MagicMock(return_value=(([1, 2]), ([3, 4]), ([5, 6]), ([7, 8])))
        mocked_fit.return_value = (self.dummy_data_df, self.dummy_data_df)

        # call the method to be tested.
        self.test_train_model.train()

        # assertions to make sure all methods are called with expected arguments.
        self.test_train_model.read_data.assert_has_calls([call('dummy_path1'), call('dummy_path2'),
                                                          call('dummy_path3'), call('dummy_path4')
                                                          ])

        self.test_train_model.transform_data.assert_has_calls([
            call(self.dummy_data_df, self.config.ALL_PRODUCT_ATTRS,
                 ['SID_IDX', 'CONFIG_ID', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE', 'BRAND'],
                 'SID_IDX'),
            call(self.dummy_data_df, self.config.ALL_PRODUCT_ATTRS,
                 ['CUSTOMER_IDX', 'CONFIG_ID', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE',
                  'BRAND'],
                 'CUSTOMER_IDX')
        ])

        mocked_fit.assert_called_once_with(([1, 2]), ([1, 2]),
                                           ([3, 4]), ([3, 4]),
                                           ([5, 6]), ([5, 6]),
                                           ([7, 8]), ([7, 8]),
                                           self.dummy_data_df,
                                           self.dummy_data_df)

        self.test_train_model.write_data.assert_called_once_with(self.dummy_data_df)
