import pandas as pd

from recommender import Recommender
from transformations import uniform_data, get_pairwise_occurance
from utils import get_tuples


class TrainModel:

    def __init__(self, config):
        self.config = config
        # Set project parameters
        self.domain_list = config.DOMAINS
        self.segment_list = config.SEGMENTS
        self.categories = config.CATEGORIES
        self.category_scores = config.CATEGORY_SCORES
        self.num_recos = config.N_RECOS
        self.training_window = config.TRAINING_WINDOW

        self.brand_filter_flag = config.BRAND_FILTER_FLAG
        self.category_filter_flag = config.CATEGORY_FILTER_FLAG
        self.data_paths = config.DATA_PATHS

        self.product_attributes = config.ALL_PRODUCT_ATTRS
        self.common_atts = config.COMMON_ATTRS
        self.model_data_path = config.MODEL_DATA_PATH

    def train(self):
        viewed_together_data = self.read_data(self.data_paths[self.config.VIEWED_TOGETHER])
        bought_together_data = self.read_data(self.data_paths[self.config.BOUGHT_TOGETHER])
        all_products_data = self.read_data(self.data_paths[self.config.ALL_PRODUCTS])
        price_list_data = self.read_data(self.data_paths[self.config.PRICE_LIST])

        """getting some columns in lower case"""
        transformed_all_products_data = uniform_data(all_products_data, self.product_attributes)

        """explode the lists into tuples of combinations per session ID for views and per user in bought"""
        print("For the view Dataframe breaking lists of brands, product categories, product_types "
              "into permutations of brands, product categories, product_types as a list of tuples")
        viewed_together_cols, group_by_col = ['SID_IDX', 'CONFIG_ID', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE',
                                              'BRAND'], 'SID_IDX'
        (tuple_list_viewed_brand,
         tuple_list_viewed_product_category,
         tuple_list_viewed_product_type,
         tuple_list_viewed_config) = self.transform_data(viewed_together_data, self.product_attributes,
                                                         viewed_together_cols,
                                                         group_by_col)

        print("For the bought Dataframe breaking lists of brands, product categories, product_types "
              "into permutations of brands, product categories, product_types as a list of tuples")

        bought_together_cols, group_by_col = ['CUSTOMER_IDX', 'CONFIG_ID', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE',
                                              'BRAND'], 'CUSTOMER_IDX'

        (tuple_list_bought_brand,
         tuple_list_bought_product_category,
         tuple_list_bought_product_type,
         tuple_list_bought_config) = self.transform_data(bought_together_data, self.product_attributes,
                                                         bought_together_cols,
                                                         group_by_col)

        recommender = Recommender()
        trained_data, _ = recommender.fit(tuple_list_viewed_brand, tuple_list_bought_brand,
                                          tuple_list_viewed_product_category,
                                          tuple_list_bought_product_category,
                                          tuple_list_viewed_product_type,
                                          tuple_list_bought_product_type,
                                          tuple_list_viewed_config,
                                          tuple_list_bought_config,
                                          transformed_all_products_data, price_list_data)
        self.write_data(trained_data)

    def read_data(self, data_path):
        print(f'reading data from path {data_path}')
        pickled_data = pd.read_pickle(data_path)

        return pickled_data

    def write_data(self, data):
        pd.to_pickle(data, self.model_data_path)

    def transform_data(self, df, prod_attrs, df_cols, df_group_by):
        uniformed_data = uniform_data(df, prod_attrs)
        data_list = get_pairwise_occurance(uniformed_data, df_cols, df_group_by)
        return get_tuples(data_list)
