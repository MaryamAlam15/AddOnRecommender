import numpy as np
import pandas as pd

from transformations import get_all_data_with_price_info
from utils import get_pairwise_bayesian_score, get_merged_score


class Recommender:

    def fit(self, tuple_list_viewed_brand, tuple_list_bought_brand,
            tuple_list_viewed_product_category, tuple_list_bought_product_category,
            tuple_list_viewed_product_type, tuple_list_bought_product_type,
            tuple_list_viewed_config, tuple_list_bought_config,
            transformed_all_products_data, price_list_data):
        """getting brand score"""
        print("Getting Pair wise bayesian score for all combinations of brands")
        """getting bayesian view score and then bought score and then combining the two"""
        grouped_view_bought_brand_count = self.get_bayesian_score_per_attr(tuple_list_viewed_brand,
                                                                           tuple_list_bought_brand,
                                                                           ["brand", "brand_cloned"])

        """get product_category score"""
        print("Getting Pair wise bayesian score for all combinations of product categories")
        """getting bayesian view score and then bought score and then combining the two"""
        grouped_view_bought_product_category_count = self.get_bayesian_score_per_attr(
            tuple_list_viewed_product_category,
            tuple_list_bought_product_category,
            ["product_category",
             "product_category_cloned"])

        """getting product_type score"""
        print("Getting Pair wise bayesian score for all combinations of product types")
        """getting bayesian view score and then bought score and then combining the two"""

        grouped_view_bought_product_type_count = self.get_bayesian_score_per_attr(tuple_list_viewed_product_type,
                                                                                  tuple_list_bought_product_type,
                                                                                  ["product_type",
                                                                                   "product_type_cloned"])

        """getting config id score"""
        print("Getting Pair wise bayesian score for all combinations of CONFIG ID's")
        grouped_view_bought_config_id_count = self.get_bayesian_score_per_attr(tuple_list_viewed_config,
                                                                               tuple_list_bought_config,
                                                                               ["product", "product_cloned"])

        """getting price % change information for all combinations of products"""
        transformed_all_products_data = transformed_all_products_data[
            ['CONFIG_ID', 'BRAND', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE', 'SERIES']].drop_duplicates(subset=['CONFIG_ID'])
        all_data = get_all_data_with_price_info(transformed_all_products_data, grouped_view_bought_config_id_count,
                                                price_list_data)

        grouped_view_bought_brand_count.reset_index(inplace=True)
        grouped_view_bought_product_category_count.reset_index(inplace=True)
        grouped_view_bought_product_type_count.reset_index(inplace=True)

        return self.get_features(all_data, grouped_view_bought_brand_count,
                                 grouped_view_bought_product_category_count,
                                 grouped_view_bought_product_type_count)

    def get_features(self, all_data, grouped_view_bought_brand_count, grouped_view_bought_product_category_count,
                     grouped_view_bought_product_type_count):
        """This function gives price percentage change for every pair of products

            Parameters
            ----------
            all_data: pd.DataFrame()
            grouped_view_bought_brand_count: pd.DataFrame()
            grouped_view_bought_product_category_count: pd.DataFrame()
            grouped_view_bought_product_type_count: pd.DataFrame()

            Returns
            -------
            pd.DataFrame(), pd.DataFrame()
        """
        """get the scores of brand, product, category, product_type along with config_id score in one dataframe"""
        print(
            "getting brand scores, product category scores, product type scores into one Dataframe for all combinations of products")
        all_data.set_index(['BRAND', 'BRAND_CLONED'], inplace=True)
        grouped_view_bought_brand_count.set_index(['BRAND', 'BRAND_CLONED'], inplace=True)
        all_data_brand = pd.merge(all_data, grouped_view_bought_brand_count, left_index=True, right_index=True,
                                  how="left")
        all_data_brand.reset_index(inplace=True)

        grouped_view_bought_product_type_count.set_index(['PRODUCT_TYPE', 'PRODUCT_TYPE_CLONED'], inplace=True)
        all_data_brand.set_index(['PRODUCT_TYPE', 'PRODUCT_TYPE_CLONED'], inplace=True)
        all_data_brand_pt = pd.merge(all_data_brand, grouped_view_bought_product_type_count, left_index=True,
                                     right_index=True)
        all_data_brand_pt.reset_index(inplace=True)

        all_data_brand_pt.set_index(['PRODUCT_CATEGORY', 'PRODUCT_CATEGORY_CLONED'], inplace=True)
        grouped_view_bought_product_category_count.set_index(['PRODUCT_CATEGORY', 'PRODUCT_CATEGORY_CLONED'],
                                                             inplace=True)
        all_data_brand_pt_pc = pd.merge(all_data_brand_pt, grouped_view_bought_product_category_count, left_index=True,
                                        right_index=True)
        all_data_brand_pt_pc.reset_index(inplace=True)
        # some experiment around ensuring every product gets 12 recommendations
        all_data_brand_pt_pc['reco_count'] = all_data_brand_pt_pc.groupby(['PRODUCT'])['PRODUCT_CATEGORY'].transform(
            'count')
        all_data_brand_pt_pc['reco_count_less'] = np.where(all_data_brand_pt_pc['reco_count'] < 12, 1, 0)
        return all_data_brand_pt_pc[all_data_brand_pt_pc['reco_count_less'] == 0], all_data_brand_pt_pc[
            all_data_brand_pt_pc['reco_count_less'] == 1].rename(
            columns={'PRODUCT': 'CONFIG_ID', 'PRODUCT_CLONED': 'PRODUCT_ID'})

    def get_bayesian_score_per_attr(self, tuple_list_viewed, tuple_list_bought, cols):
        grouped_view_count = get_pairwise_bayesian_score(tuple_list_viewed, cols, "view")
        grouped_bought_count = get_pairwise_bayesian_score(tuple_list_bought, cols, "bought")
        merged_score = get_merged_score(grouped_view_count, grouped_bought_count, cols)

        columns = {}
        for c in cols:
            columns.update({c: c.upper()})

        merged_score = merged_score.rename(columns=columns)

        return merged_score
