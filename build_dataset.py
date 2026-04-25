import pandas as pd
import os


class BuildDataset:
    def __init__(self, city, feature):
        self.city    = city.title()
        self.feature = feature.title()  

        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.file_path = os.path.normpath(os.path.join(
            base_dir, 'Web Scraping', 'CITY', self.city,
            f'car_dataset_{self.city}.json'         
        ))
        self.dataset_path  = os.path.normpath(os.path.join(
            base_dir, self.feature, '_dataset.csv'
        ))
        self.processed_log = os.path.normpath(os.path.join(
            base_dir, 'build_tracker.txt'
        ))
        if not os.path.exists(self.file_path):
            print(f"⚠️ File not found for {self.city}: {self.file_path}")
        
        if not os.path.exists(self.dataset_path ):
            print(f"⚠️ Output file not found — will be created: {self.dataset_path }")
              
        if not os.path.exists(self.processed_log):
            print(f"⚠️ build_tracker.txt not found — will be created on first run.") 

        self.df = None

    def get_required_columns(self):
        '''This function will load the required feature for given self.feature 
        from the corresponding features.txt file in the self.feature folder. '''
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(base_dir, self.feature, 'features.txt'))
        with open(path, 'r') as f:
            return [line for line in f.read().splitlines() if line.strip()]

    def is_already_processed(self): 
        with open(self.processed_log, 'r') as f:        
            processed = set(line.strip() for line in f if line.strip())         # set for O(1) lookup — RAM efficient
        return f"{self.file_path}::{self.feature}" in processed

    def mark_done(self):
        with open(self.processed_log, 'a') as f:
            f.write(f"{self.file_path}::{self.feature}\n")

    def load_data(self):
        if self.is_already_processed():
            print(f"⚠️ Already processed: {self.city} → {self.feature}")
            return False

        if not os.path.exists(self.file_path):
            print(f"❌ File not found: {self.file_path}")
            return False

        print(f"📥 Loading: {self.file_path}...")
 
        features = self.get_required_columns()
 
        try:
            if self.city == 'Ahmedabad':
                self.df = pd.read_json(self.file_path, lines=False)
            else:
                self.df = pd.read_json(self.file_path, lines=True)
        except ValueError as e:
            print(f"❌ Failed to read JSON for {self.city}: {e}")
            return False

        # drop unwanted columns from RAM
        available = [f for f in features if f in self.df.columns]
        missing   = set(features) - set(available)

        if missing:
            print(f"⚠️  Missing columns skipped: {missing}")
 
        self.df = self.df[available].copy() 
        if self.feature == 'Price' and 'city' not in self.df.columns:
            self.df['city'] = self.city
 
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"🗑️  Dropped {before - len(self.df)} duplicate rows.")
        print(f"✅ Loaded {len(self.df)} clean rows.")

        self.save_csv()
        self.mark_done()

        # explicitly free df from RAM after saving
        del self.df
        self.df = None
        return True

    def save_csv(self):
        if self.df is None or self.df.empty:
            print("⚠️  No data to save.")
            return

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        if os.path.exists(self.dataset_path): 
            existing_df = pd.read_csv(self.dataset_path, dtype=str)
            combined_df = pd.concat([existing_df, self.df.astype(str)], ignore_index=True)

            # free existing df after appending
            del existing_df, self.df
            self.df = None

            before = len(combined_df)
            combined_df.drop_duplicates(inplace=True)
            print(f"🗑️  Dropped {before - len(combined_df)} duplicates from combined data.")
            combined_df.to_csv(self.dataset_path, index=False)
            print(f"💾 Appended. Total rows: {len(combined_df)}")

            # free combined df after saving
            del combined_df

        else:
            self.df.to_csv(self.dataset_path, index=False)
            print(f"💾 Created new _dataset.csv with {len(self.df)} rows")

            # free RAM immediately
            del self.df
            self.df = None

 

if __name__ == '__main__':
    base_dir       = os.path.dirname(os.path.abspath(__file__))
    processed_file = os.path.normpath(os.path.join(base_dir, 'Web Scraping', 'processed.txt'))

    if not os.path.exists(processed_file):
        print(f"❌ processed.txt not found at: {processed_file}")
        exit()

    with open(processed_file, 'r') as f:
        CITIES = [line.strip() for line in f if line.strip()]

    if not CITIES:
        print("❌ No cities found in processed.txt")
        exit()

    FEATURES = ['Price']

    total_done    = 0
    total_skipped = 0

    for feature in FEATURES:
        print(f"\n{'='*50}")
        print(f"📂 Feature: {feature.upper()}")
        print(f"{'='*50}")

        for city in CITIES:
            print(f"\n🏙️  City: {city}")
            dataset = BuildDataset(city=city, feature=feature)
            result  = dataset.load_data()

            if result:
                print(f"✅ Done: {city} → {feature}")
                total_done += 1
            else:
                print(f"⏭️  Skipped: {city} → {feature}")
                total_skipped += 1

    print(f"\n{'='*50}")
    print(f"🎉 All done! Processed: {total_done} | Skipped: {total_skipped}")
    print(f"{'='*50}")