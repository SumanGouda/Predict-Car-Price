import json
import time
import random
from splinter import Browser
from bs4 import BeautifulSoup

class CarFastScraper:
    def __init__(self, city, links_file):
        self.json_file = f"car_dataset_{city}.json"
        self.links_file = links_file
        self.browser = None

    def _get_browser(self):
        return Browser('chrome', headless=True)

    def _close_popups(self):
        """Quick check for popups with minimal delay."""
        try:
            # Short wait_time keeps this check fast
            if self.browser.is_element_present_by_css('.cross_icon img', wait_time=0.5):
                close_btn = self.browser.find_by_css('.cross_icon img').first
                self.browser.execute_script("arguments[0].click();", close_btn._element)
                # Reduced wait after closing
                time.sleep(0.2) 
        except:
            pass

    def _click_logic(self):
        try:
            # Wait up to 2s for button (reduced from 3s)
            if self.browser.is_element_present_by_text('View all Specifications', wait_time=2):
                btn = self.browser.find_by_text('View all Specifications').first
                self.browser.execute_script("arguments[0].scrollIntoView();", btn._element)
                self.browser.execute_script("arguments[0].click();", btn._element)
                
                # Dynamic check: Did the button disappear? 
                # If it's gone, the specs expanded and we can move on immediately.
                if self.browser.is_element_not_present_by_text('View all Specifications', wait_time=1.0):
                    print("   [Log] Specs expanded fast.")
                    return True
                
                # Only if still present do we trigger the "stubborn" wait
                print("   [Wait] Specs slow. Giving heavy assets 4s...")
                time.sleep(4.0) 
                self.browser.execute_script("arguments[0].click();", btn._element)
                time.sleep(0.5)
                return True
        except:
            pass
        return False

    def _extract_data(self, soup, url):
        title = soup.title.string if soup.title else ""
        if "used cars for sale" in title.lower() or "Price" not in str(soup):
            return None

        data = {"url": url, "car_name": "Unknown", "Price": "N/A"}
        name_tag = soup.find('div', class_='vehicleName')
        h1 = name_tag.find('h1') if (name_tag and name_tag.find('h1')) else soup.find('h1')
        if h1:
            parts = h1.get_text(separator="|", strip=True).split("|")
            data["car_name"] = parts[1].strip() if len(parts) >= 2 else parts[0].strip()

        price_div = soup.find('div', class_='vehiclePrice') or soup.find('span', class_='amount')
        if price_div:
            price_text = price_div.find('span') or price_div
            data["Price"] = price_text.get_text(strip=True)

        for item in soup.find_all('li', class_='gsc_col-xs-12'):
            label = item.find('div', class_='label')
            val = item.find('span', class_='value-text')
            if label and val:
                data[label.get_text(strip=True)] = val.get_text(strip=True)
        return data

    def save_to_json(self, data, count):
        with open(self.json_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(data) + "\n")
        print(f"✅ Saved Record #{count}: {data.get('car_name')}")

    def scrape_all(self):
        with open(self.links_file, "r") as f:
            all_links = [line.strip() for line in f.readlines()]

        self.browser = self._get_browser()
        self.browser.driver.set_window_size(1600, 900)

        for index, link in enumerate(all_links):
            current_count = index + 1
            if not link or "used-car-details" not in link: continue

            try:
                self.browser.visit(link)
                
                # Check for popups immediately with 0.5s timeout
                self._close_popups()
                
                # Dynamic wait for H1 instead of hard sleep
                if not self.browser.is_element_present_by_css('h1', wait_time=4):
                    self.browser.visit(link)

                self.browser.execute_script("window.scrollTo(0, 500);")
                self._click_logic()

                # Parse the HTML immediately after click logic returns
                page_soup = BeautifulSoup(self.browser.html, 'html.parser')
                car_data = self._extract_data(page_soup, link)

                if car_data:
                    self.save_to_json(car_data, current_count)
                else:
                    print(f"❌ Record #{current_count} skipped (Sold/Listing).")

            except Exception as e:
                print(f"⚠️ Error at #{current_count}: {e}")
                self.browser.quit()
                self.browser = self._get_browser()
                continue

        self.browser.quit()
        
# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    target_city = "Ahmedabad"  
    links_filename = r"D:\IMP  ML  PROJECTS\CAR PRICE PREDICTION\web scraping\extraction\car_links_ahmedabad.txt"  
    
    scraper = CarFastScraper(city=target_city, links_file=links_filename)
    
    print(f"🚀 Starting scraper for {target_city}...")
    print(f"📂 Data will be saved to: car_dataset_{target_city}.json")
    print("-" * 30)
    
    try:
        scraper.scrape_all()
        print("-" * 30)
        print("✅ Task finished successfully.")
    except KeyboardInterrupt:
        print("\n🛑 Scraper stopped manually by user.")
    except Exception as e:
        print(f"\n❌ A fatal error occurred: {e}")