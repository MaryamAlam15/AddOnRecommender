import json

import pandas as pd

from config import AddOnProductsConfig
from transformations import convert_result_to_json

pd.set_option("display.max_rows", 500, "display.max_columns", 100)
pd.set_option('display.max_colwidth', 500)
pd.options.mode.chained_assignment = None


def load_model(path):
    return pd.read_pickle(path)


def predict(filter_list, filter_score, all_data, num_recos):
    df = [pd.DataFrame()] * len(filter_list)

    for index, c in enumerate(filter_list):
        df[index] = predict_by_filter(c, all_data, filter_score, num_recos)
        df[index]["tmp"] = 1
        df[index]["ranking"] = df[index].sort_values(["PRODUCT", "final_score"],
                                                     ascending=False).groupby("PRODUCT")["tmp"].cumcount() + 1

        df[index] = df[index][df[index].ranking <= num_recos].rename(columns={'PRODUCT': 'CONFIG_ID',
                                                                              'PRODUCT_CLONED': 'PRODUCT_ID'})[
            ["CONFIG_ID", "PRODUCT_ID", "ranking"]]

    return df


def predict_by_filter(filter_, all_data, category_scores, num_recos, brand_filter=False):
    """This function filters dataframes by categories and calculates final scores based on
    brand_score, product_category_score and product_type_score as per config variable

           Parameters
           ----------
           filter_: list
           all_data: pd.DataFrame()
           category_scores: dict
           num_recos: int
           brand_filter: Boolean

           Returns
           -------
           pd.DataFrame()
    """
    print("predicting for filter %s" % (filter_))

    if brand_filter == '1':
        df = all_data[all_data["BRAND"] == filter_]
        df['brand_score'] = 0
    else:
        df = all_data[all_data["PRODUCT_CATEGORY"] == filter_]

        df.info()

        df['brand_score'] = (df['brand_score'] - df['brand_score'].min()) / df['brand_score'].max()

    df['product_score'] = (df['product_score'] - df['product_score'].min()) / df['product_score'].max()
    df['product_category_score'] = (df['product_category_score'] - df['product_category_score'].min()) / df[
        'product_category_score'].max()
    df['product_type_score'] = (df['product_type_score'] - df['product_type_score'].min()) / df[
        'product_type_score'].max()

    df['final_score'] = category_scores[filter_]["PRODUCT"] * df['product_score'] + category_scores[filter_]["BRAND"] * \
                        df['brand_score'] + \
                        category_scores[filter_]["PRODUCT_CATEGORY"] * df['product_category_score'] + \
                        category_scores[filter_]["PRODUCT_TYPE"] * df['product_type_score'] \
                        - category_scores[filter_]["PRICING_SCORE"] * df['percentage_price_change']

    df_final = df.groupby(['PRODUCT']).apply(
        lambda x: x.sort_values(['final_score'], ascending=False)[:num_recos]).reset_index(drop=True)
    df_final['ranking'] = df_final[['PRODUCT', 'PRODUCT_CLONED']].groupby(['PRODUCT'])['PRODUCT_CLONED'].cumcount() + 1
    return df_final


def get_json_result(path, df_list, categories, segment_list, domain_list):
    final_json = []

    for i in range(0, len(df_list)):
        final_json.append(convert_result_to_json(df_list[i], categories, segment_list, domain_list))

    with open(path, 'w+') as outfile:
        json.dump(final_json, outfile)


if __name__ == '__main__':
    config = AddOnProductsConfig()

    model_data = load_model(config.MODEL_DATA_PATH)

    recommended_data = predict(config.CATEGORIES, config.CATEGORY_SCORES, model_data, config.N_RECOS)
    get_json_result(config.PREDICTED_DATA_PATH, recommended_data, config.CATEGORIES, config.SEGMENTS, config.DOMAINS)
