
from data_collection_api import main as collect_main
from data_wrangling import main as wrangle_main
from eda_charts import main as eda_main
from model_training import main as train_main

if __name__ == "__main__":
    collect_main()
    wrangle_main()
    eda_main()
    train_main()
    print("\n✅ All steps completed. You can now run: python spacex_dash_app.py")
