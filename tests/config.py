
class Config:
    # For which domains will a recommendation be computed
    DOMAINS = ["DE", "AT"]

    # Tabs to be displayed for each segment
    CATEGORIES = ["parfum", "make-up", "pflege", "haare"]

    CATEGORY_SCORES = {
        "parfum": {"PRODUCT": 0.5, "BRAND": 0, "PRODUCT_TYPE": 0, "PRODUCT_CATEGORY": 0.5, "PRICING_SCORE": 0},
        "make-up": {"PRODUCT": 0.5, "BRAND": 0.25, "PRODUCT_TYPE": 0, "PRODUCT_CATEGORY": 0, "PRICING_SCORE": 0.25},
        "pflege": {"PRODUCT": 0.5, "BRAND": 0, "PRODUCT_TYPE": 0.15, "PRODUCT_CATEGORY": 0.15, "PRICING_SCORE": 0.2},
        "haare": {"PRODUCT": 0.5, "BRAND": 0, "PRODUCT_TYPE": 0.15, "PRODUCT_CATEGORY": 0.15, "PRICING_SCORE": 0.2}
    }

    # BRAND_SCORES = {
    #    "chanel": {"PRODUCT": 0.5, "BRAND": 0, "PRODUCT_TYPE": 0.1, "PRODUCT_CATEGORY": 0.2, "PRICING_SCORE": 0.2}
    # }

    SEGMENTS = ["Default", "Premium", "Non-Premium"]

    # Number of recommendations per product
    N_RECOS = 5

    # Training Window
    TRAINING_WINDOW = 1

    BRAND_FILTER_FLAG = '0'
    CATEGORY_FILTER_FLAG = '1'

    VIEWED_TOGETHER = 'viewed_together'
    BOUGHT_TOGETHER = 'bought_together'
    ALL_PRODUCTS = 'all_products'
    PRICE_LIST = 'price_list'

    DATA_PATHS = {
        VIEWED_TOGETHER: 'dummy_path1',
        BOUGHT_TOGETHER: 'dummy_path2',
        ALL_PRODUCTS: 'dummy_path3',
        PRICE_LIST: 'dummy_path4'
    }

    MODEL_DATA_PATH = 'model_data/recommender_model.pkl'

    ALL_PRODUCT_ATTRS = ['BRAND', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE']
    COMMON_ATTRS = ALL_PRODUCT_ATTRS + ['CONFIG_ID']