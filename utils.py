import itertools

import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)


def add_to_tuple(col, tuple_list):
    """A helper function to calculate pairwise permutations of products

           Parameters
           ----------
           l: pd.Series
           tuple_list: [()]  - list of tuples

           Returns
           -------
           [()]
    """
    for pair in itertools.permutations(col, 2):
        tuple_list.append(pair)


def get_pairwise_bayesian_score(tuple_list, columns, type):
    """This function calculates grouped brand/product_type/product_category/product,
    brand_cloned/product_type_cloned/product_category_cloned/product_cloned counts
    and associates a score for every combination of brand/product_type/product_category/product
    through bayesian score
        Parameters
        ----------
        tuple_list: [()]
        columns = list
        type = str
        Returns
        -------
        pd.DataFrame()
    """
    grouped_count = pd.DataFrame(tuple_list, columns=columns)
    grouped_count['dummy'] = 1

    grouped_count['count'] = grouped_count.groupby(columns).transform('count')

    grouped_count[columns[0] + '_count'] = grouped_count.groupby(columns[0])[columns[1]].transform('count')
    grouped_count[columns[1] + '_count'] = grouped_count.groupby(columns[1])[columns[0]].transform('count')

    grouped_count['not_' + columns[0] + '_not_' + columns[1]] = len(grouped_count) - grouped_count[
        columns[0] + '_count'] - grouped_count[columns[1] + '_count'] + grouped_count[columns[0] + '_count']
    grouped_count['not_' + columns[1]] = len(grouped_count) - grouped_count[columns[1] + '_count']

    grouped_count[type + '_' + columns[0] + '_score'] = (grouped_count['count'] / grouped_count[
        columns[1] + '_count']) / ((
                grouped_count['not_' + columns[0] + '_not_' + columns[1]] / grouped_count['not_' + columns[1]]))
    grouped_count = grouped_count[columns + [type + '_' + columns[0] + '_score']]
    return grouped_count


def get_merged_score(grouped_view_count, grouped_bought_count, columns):
    """This function makes tuples

            Parameters
            ----------
            data_list: pd.DataFrame()

            Returns
            -------
            tuple_list_brand: [()]
            tuple_list_product_category: [()]
            tuple_list_product_type: [()]
            tuple_list_config: [()]
    """
    grouped_view_count.drop_duplicates(subset=columns + ['view_' + columns[0] + '_score'], inplace=True)
    grouped_view_count.set_index(columns, inplace=True)
    grouped_bought_count.drop_duplicates(subset=columns + ['bought_' + columns[0] + '_score'], inplace=True)
    grouped_bought_count.set_index(columns, inplace=True)
    grouped_view_bought_count = pd.merge(grouped_view_count, grouped_bought_count, left_index=True, right_index=True,
                                         how="left").fillna(0)
    grouped_view_bought_count.reset_index(inplace=True)
    grouped_view_bought_count[columns[0].lower() + '_score'] = 0.8 * grouped_view_bought_count[
        'view_' + columns[0] + '_score'] + 0.2 * grouped_view_bought_count['bought_' + columns[0] + '_score']
    grouped_view_bought_count = grouped_view_bought_count[columns + [columns[0].lower() + '_score']]
    return grouped_view_bought_count


def get_tuples(data_list):
    """This function makes tuples

        Parameters
        ----------
        data_list: pd.DataFrame()

        Returns
        -------
        tuple_list_brand: [()]
        tuple_list_product_category: [()]
        tuple_list_product_type: [()]
        tuple_list_config: [()]
    """
    tuple_list_config = []
    tuple_list_brand = []
    tuple_list_product_type = []
    tuple_list_product_category = []

    data_list.apply(lambda x: add_to_tuple(x['CONFIG_ID'], tuple_list_config), axis=1)
    data_list.apply(lambda x: add_to_tuple(x['BRAND'], tuple_list_brand), axis=1)
    data_list.apply(lambda x: add_to_tuple(x['PRODUCT_TYPE'], tuple_list_product_type), axis=1)
    data_list.apply(lambda x: add_to_tuple(x['PRODUCT_CATEGORY'], tuple_list_product_category), axis=1)

    return tuple_list_brand, tuple_list_product_category, tuple_list_product_type, tuple_list_config