import pandas as pd
import os

class BuildDataset:
    def __init__(self, city, feature):
        self.city = city
        self.feature = feature.lower()

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.file_path = os.path.normpath(os.path.join(
            base_dir, 'web scraping', city, f'car_dataset_{city.lower()}.json'
        ))
        self.output_path = os.path.normpath(os.path.join(
            base_dir, self.feature, '_dataset.csv'
        ))
        self.done_file = os.path.normpath(os.path.join(
            base_dir, 'processed_files.txt'
        ))
        self.df = None

    def load_feature(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(base_dir, self.feature, 'features.txt'))
        with open(path, 'r') as f:
            return [line for line in f.read().splitlines() if line.strip()]
        
    def is_already_processed(self):
        if not os.path.exists(self.done_file):
            return False
        with open(self.done_file, 'r') as f:
            return f"{self.file_path}::{self.feature}" in f.read().splitlines()

    def make_done(self):
        with open(self.done_file, 'a') as f:
            f.write(f"{self.file_path}::{self.feature}\n")   

    def load_data(self):
        if self.is_already_processed():
            print(f"⚠️  Already processed: {self.city} → {self.feature}")
            return False

        if not os.path.exists(self.file_path):
            print(f"❌ File not found: {self.file_path}")
            return False

        print(f"📥 Loading: {self.file_path}...")
        features = self.load_feature()
        self.df = pd.read_json(self.file_path, lines=True)

        # ✅ filter to only existing columns
        available = [f for f in features if f in self.df.columns]
        missing = set(features) - set(available)
        if missing:
            print(f"⚠️  Missing columns skipped: {missing}")
        self.df = self.df[available]

        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"🗑️  Dropped {before - len(self.df)} duplicate rows.")
        print(f"✅ Loaded {len(self.df)} clean rows.")

        self.save_csv()
        self.make_done()
        return True

    def save_csv(self):
        if self.df is None or self.df.empty:
            print("⚠️  No data to save.")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if os.path.exists(self.output_path):
            existing_df = pd.read_csv(self.output_path)
            combined_df = pd.concat([existing_df, self.df], ignore_index=True)
            before = len(combined_df)
            combined_df.drop_duplicates(inplace=True)
            print(f"🗑️  Dropped {before - len(combined_df)} duplicates from combined data.")
            combined_df.to_csv(self.output_path, index=False)
            print(f"💾 Appended. Total rows: {len(combined_df)}")
        else:
            self.df.to_csv(self.output_path, index=False)
            print(f"💾 Created new _dataset.csv with {len(self.df)} rows")

        del self.df
        self.df = None


# ─────────────────────────────────────────
# ✅ Run for all features and cities
# ─────────────────────────────────────────
if __name__ == '__main__':

    FEATURES = ['mileage', 'power', 'price']
    CITIES = []

    for feature in FEATURES:
        print(f"\n{'='*50}")
        print(f"📂 Feature: {feature.upper()}")
        print(f"{'='*50}")

        for city in CITIES:
            print(f"\n🏙️  City: {city}")
            dataset = BuildDataset(city=city, feature=feature)
            result = dataset.load_data()

            if result:
                print(f"✅ Done: {city} → {feature}")
            else:
                print(f"⏭️  Skipped: {city} → {feature}")

    print("\n🎉 All done!")
