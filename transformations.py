import json

import numpy as np
import pandas as pd

pd.set_option('display.max_colwidth', 500)


def apply_segments(x, indent_list):
    y = {}
    for d in indent_list:
        y[d] = x
    return y


def get_coverage_df(viewed_together_data, all_products_data, col):
    """Calculates the products not present in the training period

        Parameters
        ----------
        viewed_together_data: pd.DataFrame()
        all_products_data: pd.DataFrame()

        Returns
        -------
        pd.DataFrame()
    """
    df1 = viewed_together_data[[col]].drop_duplicates()
    df2 = all_products_data[[col]].drop_duplicates()

    df1.set_index([col], inplace=True)
    df2.set_index([col], inplace=True)

    df3 = df2[~df2.index.isin(df1.index)]
    coverage_prods = pd.merge(all_products_data, df3, on=col, how="inner")

    print("Returning list of products not included in the training period. Will be used later for 100% coverage"
          " calculation")
    return coverage_prods


def uniform_data(df, columns):
    """A helper function to change the Brand, Product Category and Product Type columns to lower case

        Parameters
        ----------
        df: pd.DataFrame()

        Returns
        -------
        pd.DataFrame()
    """
    for col in columns:
        # remove white space n convert the value to lower.
        df[col] = df[col].str.replace(' ', '').str.lower()

    print("Returning dataframe with lower cases for brand, product_category and product_type")
    return df


def get_pairwise_occurance(df, columns_list, group_by_col):
    data_list = df[columns_list].drop_duplicates()

    print(f"Aggregating product_category, product_type and brand as a list at the {group_by_col} level for dataframe")
    return pd.DataFrame(
        data_list[columns_list].groupby(group_by_col).agg(lambda x: x.tolist())).reset_index()


def get_all_data_with_price_info(all_prod_data, grouped_view_bought_df, price_list_data):
    """This function gives price percentage change for every pair of products

        Parameters
        ----------
        all_prod_data: pd.DataFrame()
        grouped_view_bought_df: pd.DataFrame()
        price_list_data: pd.DataFrame()

        Returns
        -------
        pd.DataFrame()
    """
    price_list_data['PRICE_PER_UNIT'] = price_list_data['PRICE_PER_UNIT'].astype(float)
    print("Calculating % price difference for all combinations of products")
    product_to_features = all_prod_data.copy()
    product_to_features.set_index(['CONFIG_ID'], inplace=True)
    price_list_data.set_index(['CONFIG_ID'], inplace=True)
    price_list_data.columns = ['PRICE_PER_UNIT']

    grouped_view_bought_df = grouped_view_bought_df.rename(columns={'PRODUCT': 'CONFIG_ID'})
    grouped_view_bought_df.set_index(['CONFIG_ID'], inplace=True)
    product_data_without_price = pd.merge(grouped_view_bought_df, product_to_features, left_index=True,
                                          right_index=True)
    product_data_with_price = pd.merge(product_data_without_price, price_list_data, left_index=True, right_index=True)

    product_data_with_price.reset_index(inplace=True)
    product_data = product_data_with_price.rename(columns={'CONFIG_ID': 'PRODUCT', 'PRODUCT_CLONED': 'CONFIG_ID'})
    product_data.set_index(['CONFIG_ID'], inplace=True)
    product_to_features.columns = ["BRAND_CLONED", "PRODUCT_CATEGORY_CLONED", "PRODUCT_TYPE_CLONED", "SERIES_CLONED"]
    price_list_data.columns = ['PRICE_PER_UNIT_CLONED']
    all_data_without_price = pd.merge(product_data, product_to_features, left_index=True, right_index=True)
    all_data_with_price = pd.merge(all_data_without_price, price_list_data, left_index=True, right_index=True)

    all_data_with_price.reset_index(inplace=True)
    allData = all_data_with_price.rename(columns={'CONFIG_ID': 'PRODUCT_CLONED'})
    allData['percentage_price_change'] = (allData[['PRICE_PER_UNIT', 'PRICE_PER_UNIT_CLONED']].max(axis=1) - allData[
        ['PRICE_PER_UNIT', 'PRICE_PER_UNIT_CLONED']].min(axis=1)) / allData[
                                             ['PRICE_PER_UNIT', 'PRICE_PER_UNIT_CLONED']].max(axis=1)
    allData['price_range'] = np.where(allData['percentage_price_change'] <= 0.15, 1, 0)
    return allData


def convert_result_to_json(final_df, categories, segment_list, domain_list):
    """converts the dataframe into json

        Parameters
        ----------
        final_df: pd.DataFrame()
        categories: []
        segment_list: []
        domain_list: []

        Returns
        -------
        final_json: json
    """
    final_df_grouped = final_df.copy()
    # final_df_grouped = final_df_grouped[final_df_grouped['CONFIG_ID']=='30102888']
    final_df_grouped['PRODUCT_RANK_DICT'] = [{"PRODUCT_ID": key, "RANK": str(value)} for key, value in
                                             zip(final_df_grouped.PRODUCT_ID.values.tolist(),
                                                 final_df_grouped.ranking.values.tolist())]
    final_df_grouped = final_df_grouped.groupby(['CONFIG_ID'])[['PRODUCT_RANK_DICT']].agg(
        lambda x: x.tolist()).reset_index()

    final_df_grouped['PRODUCT_RANK_DICT'] = final_df_grouped['PRODUCT_RANK_DICT'].apply(
        lambda x: apply_segments(x, categories))
    final_df_grouped['PRODUCT_RANK_DICT'] = final_df_grouped['PRODUCT_RANK_DICT'].apply(
        lambda x: apply_segments(x, segment_list))
    final_df_grouped['PRODUCT_RANK_DICT'] = final_df_grouped['PRODUCT_RANK_DICT'].apply(
        lambda x: apply_segments(x, domain_list))

    final_list = []
    for index, row in final_df_grouped.iterrows():
        final_dict = {}
        final_dict[row['CONFIG_ID']] = row['PRODUCT_RANK_DICT']
        final_list.append(json.dumps(final_dict))

    return final_list
