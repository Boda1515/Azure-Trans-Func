from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd
import unicodedata
import numpy as np
import requests
import logging
import json
import re

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_data(data, region):
    # Convert to DataFrame
    dff = pd.json_normalize(data)
    region_cur_name = region

    # create column mapping for handling different regions CA, US, AU, JP:

    def create_column_mapping():
        """
        Create a mapping dictionary for variant column names across regions
        """
        return {
            # Standard name: [possible variant names]
            'date_column': ['date_column'],
            'product_url': ['product_url'],
            'site': ['site'],
            'category': ['category'],
            'brand': ['Brand', 'Brand Name'],
            'model_name': ['Model', 'Model Name', 'Model name'],
            'product_title': ['Title'],
            'price': ['Price'],
            'os': ['Operating System', 'Operating system', 'OS'],
            'ram_gb': ['RAM memory installed size', 'Ram Memory Installed Size', 'RAM Memory Installed', 'RAM'],
            'storage': ['Memory Storage Capacity', 'Memory storage capacity', 'Digital Storage Capacity', 'Digital storage capacity'],
            'screen_size_in': ['Screen Size', 'Screen size', 'Standing screen display size'],
            'resolution': ['Resolution', 'Display Resolution'],
            'refresh_rate_hz': ['Refresh Rate', 'Refresh rate'],
            'cpu_speed_ghz': ['CPU Speed', 'CPU speed', 'Processor Speed'],
            'connectivity_technology': ['Connectivity Technology', 'Connectivity technologies', 'Wireless communication technologies', 'Connectivity technology'],
            'cpu_model': ['CPU Model', 'CPU model', 'Processor Type'],
            'color': ['Color', 'Colour'],
            'wireless_carrier': ['Wireless Carrier', 'Wireless Provider', 'Wireless carrier'],
            'cellular_technology': ['Cellular Technology', 'Cellular technology'],
            'reviews': ['reviews', 'Customer Reviews'],
            'rate': ['Rate'],
            'discount': ['Discount'],
            'image_url': ['Image URL'],
            'asin': ['ASIN'],
            'batteries': ['Batteries'],
            'model_number': ['Item model number', 'Item Model Number', 'Model Number']
        }

    def get_first_valid_column(df, columns):
        """
        Get the first column that exists in the DataFrame from the list of columns
        """
        for col in columns:
            if col in df.columns:
                return df[col]
        return pd.Series([np.nan] * len(df))

    def standardize_columns(df):
        """
        Standardize column names across different regions and handle duplicate columns.
        """
        # Create a new DataFrame to store results
        result = pd.DataFrame(index=df.index)

        # Get the mapping
        mapping = create_column_mapping()

        # Process each standard column
        for std_name, variants in mapping.items():
            # Get existing columns that match any of the variants
            existing_variants = [col for col in variants if col in df.columns]

            if existing_variants:
                # Take the first non-null value across all variant columns
                result[std_name] = pd.NA
                for variant in existing_variants:
                    mask = result[std_name].isna()
                    result.loc[mask, std_name] = df.loc[mask, variant]

        # Add any remaining columns that weren't in the mapping
        mapped_cols = [col for variants in mapping.values()
                       for col in variants]
        remaining_cols = [col for col in df.columns if col not in mapped_cols]

        for col in remaining_cols:
            result[col] = df[col]

        # Ensure all main columns are present
        main_columns = ['date_column', 'site', 'category', 'brand', 'model_name', 'product_title', 'price', 'os',
                        'ram_gb', 'storage', 'screen_size_in', 'resolution', 'refresh_rate_hz', 'cpu_speed_ghz',
                        'connectivity_technology', 'cpu_model', 'color', 'wireless_carrier', 'cellular_technology',
                        'reviews', 'rate', 'discount', 'product_url', 'image_url', 'asin']

        for col in main_columns:
            if col not in result.columns:
                result[col] = np.nan  # Create missing columns with NaN values

        return result

    df_new = standardize_columns(dff)

    logging.info(df_new.shape)
    logging.info('-' * 50)

    # Select relevant columns
    df_new = df_new[['date_column', 'site', 'category', 'brand', 'model_name', 'product_title', 'price', 'os',
                    'ram_gb', 'storage', 'screen_size_in',
                     'resolution', 'refresh_rate_hz', 'cpu_speed_ghz', 'connectivity_technology',
                     'cpu_model', 'color', 'wireless_carrier',
                     'cellular_technology', 'reviews', 'rate', 'discount', 'product_url', 'image_url', 'asin']]

    # Data Cleaning

    # Filter the DataFrame to drop rows where 'product_title' is NaN
    df_new = df_new[~df_new['product_title'].isna()]

    # Convert all columns to lowercase except 'product_url', 'image_url'
    # List of columns to exclude from conversion
    exclude_columns = ['product_url', 'image_url']

    # Get the columns to convert
    cols_to_convert = df_new.columns.difference(exclude_columns)

    # Convert all columns to lowercase except for the excluded ones
    df_new.loc[:, cols_to_convert] = df_new.loc[:, cols_to_convert].apply(
        lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))

    # Fill Missing Data from Product Title:
    # Before
    # List of columns to check for missing values
    columns_to_check = ['brand', 'model_name', 'product_title', 'price', 'os',
                        'ram_gb', 'storage', 'screen_size_in', 'resolution',
                        'refresh_rate_hz', 'cpu_speed_ghz', 'connectivity_technology',
                        'cpu_model', 'color', 'wireless_carrier', 'cellular_technology']

    # Loop through each column and print the count of missing values
    for column in columns_to_check:
        logging.info(f'{column}: {df_new[column].isna().sum()}')

    logging.info('-' * 50)

    # Extract Brand

    def fill_brand_from_title(df, brand_column='brand', title_column='product_title'):
        # Function to extract the first word from the title
        def extract_first_word(title):
            if pd.isna(title):
                return None
            # Split the title by spaces and return the first word, converted to lowercase
            return title.split()[0].lower()

        # Apply the function to fill the missing brands
        df[brand_column] = df.apply(lambda row: extract_first_word(
            row[title_column]) if pd.isna(row[brand_column]) else row[brand_column], axis=1)

        return df

    df_new = fill_brand_from_title(df_new)

    # Extract Model Name

    def extract_model_name(title, brand=None):
        """
        Extract the model name from a smartphone title with or without brand information.

        Args:
            title (str): The full product title
            brand (str, optional): The brand name of the phone

        Returns:
            str: Extracted model name or None if no match found
        """
        # Convert input to string and clean it
        title = str(title).strip()

        # Common model patterns
        common_patterns = [
            # Phone model patterns with numbers and letters
            # Generic model numbers like SH-53A, SO-41B
            r'(?:Model\s)?([A-Z0-9]+(?:-[A-Z0-9]+)+)',
            # Models like C36, C57PRO
            r'([A-Z]{2,}[0-9]+(?:\s?[A-Z]+)?)',

            # Specific manufacturer patterns
            # Apple
            r'iPhone\s(?:SE\s)?(\d{1,2}(?:\sPlus|\sPro(?:\sMax)?)?)',

            # Sony
            r'Xperia\s([^\s]+(?:\s(?:III|II|I|Plus|Pro|Ultra|Ace|compact))?)',
            r'(?:SO|SOV)-?(\d{2}[A-Z]?)',

            # Sharp
            r'AQUOS\s([^\s]+(?:\s(?:R|sense|Zero|compact))?(?:\d)?)',
            r'SH-?(\d{2}[A-Z]?)',

            # OUKITEL
            r'OUKITEL\s([A-Z0-9]+(?:\s?PRO)?)',

            # Xiaomi/Redmi
            r'(?:Redmi\s)?(?:Note\s)?(\d{1,2}(?:\s?Pro\+?|\s?Lite)?)',
            r'Mi\s(\d{1,2}(?:\s?Lite)?(?:\s?[A-Z]+)?)',

            # Google
            r'Pixel\s(\d[a-zA-Z]?)',

            # ASUS
            r'ZenFone\s(\d(?:\sMax|\sLite)?)',
            r'ZB(\d{3}[A-Z]{2})',

            # General patterns for model numbers
            r'([A-Z0-9]{2,}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)?)',
            r'([A-Z]{1,2}[0-9]{2,}(?:[A-Z]{1,2})?)'
        ]

        # Clean the title
        title = title.replace('SIM Free', '').replace('SIM-Free', '')
        title = title.replace('Smartphone', '').strip()

        # Try each pattern
        for pattern in common_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                model = match.group(1).strip()

                # Clean up the extracted model
                model = re.sub(r'\s+', ' ', model)  # Remove extra spaces
                model = model.strip()

                # Verify the model isn't just a number or too short
                if len(model) < 2 or model.isdigit():
                    continue

                return model

        # If no patterns match, try to extract any alphanumeric sequence that might be a model
        fallback_pattern = r'([A-Z0-9]{2,}(?:-[A-Z0-9]+)?)'
        matches = re.findall(fallback_pattern, title)
        if matches:
            # Filter out common words and numbers
            filtered_matches = [m for m in matches
                                if len(m) >= 2
                                and not m.lower() in ['gb', 'ram', 'rom', 'sim', 'tb']]
            if filtered_matches:
                return filtered_matches[0]

        return None

    # Apply the function to fill null values in the 'model_name' column
    df_new['model_name'] = df_new['model_name'].fillna(
        df_new['product_title'].apply(extract_model_name))
    df_new['model_name'] = df_new['model_name'].astype(str).str.lower()

    # Extract RAM & ROM

    # Function to extract RAM and Storage

    def extract_memory_capacity(row):
        # Ensure the required columns exist
        if 'product_title' not in row or 'storage' not in row or 'ram_gb' not in row:
            return row

        title = row['product_title'].lower()

        # Find all 'number GB/gb' in the title
        matches = re.findall(r'(\d+)\s*(GB|gb)', title)

        # Initialize variables for RAM and storage
        ram_capacity = None
        storage_capacity = None

        # Process each match found
        for match in matches:
            value, unit = match  # Get the value and unit
            # Construct the full capacity string
            capacity_str = f"{value}{unit}"

            # Check for RAM
            if 'ram' in title or 'memory' in title:
                ram_context_match = re.search(
                    r'(\d+)\s*(GB|gb)\s*(ram|memory)', title)
                if ram_context_match and capacity_str == f"{ram_context_match.group(1)}{ram_context_match.group(2)}":
                    ram_capacity = capacity_str

            # Check for Storage (storage/rom)
            if 'storage' in title or 'rom' in title:
                storage_context_match = re.search(
                    r'(\d+)\s*(GB|gb)\s*(storage|rom)', title)
                if storage_context_match and capacity_str == f"{storage_context_match.group(1)}{storage_context_match.group(2)}":
                    storage_capacity = capacity_str

            # Make educated guesses if no explicit mention is found
            if ram_capacity is None and int(value) < 16:
                ram_capacity = capacity_str
            elif storage_capacity is None and int(value) >= 16:
                storage_capacity = capacity_str

        # Update the 'storage' and 'ram_gb' columns if they are null and corresponding values are found
        if pd.isnull(row['storage']) and storage_capacity:
            row['storage'] = storage_capacity

        if pd.isnull(row['ram_gb']) and ram_capacity:
            row['ram_gb'] = ram_capacity

        return row

    # Apply the function to all rows
    df_new = df_new.apply(extract_memory_capacity, axis=1)

    # Extract OS

    # Define a list of popular operating systems
    popular_os = [
        "android",
        "ios",
        "miui",
        "oxygenos",
        "one ui",
        "coloros",
        "nucleus os",
        "kaios",
        "hyper os",
        "linux",
        "blackberry",
    ]

    def extract_os(title):
        # Check if the title is None
        if not title:
            return np.nan
        # Convert title to lowercase for uniformity
        title = title.lower()
        # Find all matches of popular OS in the title
        matched_os = [os for os in popular_os if re.search(
            r'\b' + re.escape(os) + r'\b', title)]

        # Return the matched OS or None if no match is found
        return matched_os[0] if matched_os else np.nan

    # Apply the function to fill null values in the 'os' column
    df_new['os'] = df_new['os'].fillna(
        df_new['product_title'].apply(extract_os))
    df_new['os'] = df_new['os'].astype(str).str.lower()

    # Extract Screen Size

    def extract_screen_size(title):
        # Check if the title is None
        if not title:
            return np.nan
        # Convert title to lowercase for uniformity
        title = title.lower()

        # Regular expression to match patterns like '6.5 inch', '7 inches', '6-inch', etc.
        screen_size_match = re.search(
            r'(\d+(\.\d+)?)\s*(-)?\s*inch(es)?', title)

        # Return the screen size if found, otherwise return NaN
        if screen_size_match:
            # Extract the numeric part
            return float(screen_size_match.group(1))
        else:
            return np.nan

    # Apply the function to fill null values in the 'screen_size' column
    df_new['screen_size_in'] = df_new['screen_size_in'].fillna(
        df_new['product_title'].apply(extract_screen_size))
    df_new['screen_size_in'] = df_new['screen_size_in'].astype(str).str.lower()

    # Extract Resolution

    def extract_resolution(title):
        # Check if the title is None
        if not title:
            return np.nan
        # Convert title to lowercase for uniformity
        title = title.lower()

        # Regular expression to match resolution patterns like '1080x2400', '1440 × 2560', etc.
        resolution_match = re.search(r'(\d{3,4})\s*[x×]\s*(\d{3,4})', title)

        # Return the resolution if found, otherwise return NaN
        if resolution_match:
            # Extract both parts and format as width x height
            return f"{resolution_match.group(1)}x{resolution_match.group(2)}"
        else:
            return np.nan

    # Apply the function to fill null values in the 'resolution' column
    df_new['resolution'] = df_new['resolution'].fillna(
        df_new['product_title'].apply(extract_resolution))
    df_new['resolution'] = df_new['resolution'].astype(str).str.lower()

    # Extract Refresh Rate

    def extract_refresh_rate(title):
        # Check if the title is None
        if not title:
            return np.nan
        # Convert title to lowercase for uniformity
        title = title.lower()

        # Regular expression to match refresh rate patterns like '90hz', '120hz', or '144 hertz'
        refresh_rate_match = re.search(r'(\d{2,3})\s*(hz|hertz)', title)

        # Return the refresh rate if found, otherwise return NaN
        if refresh_rate_match:
            # Return the refresh rate in 'XXHz' format
            return f"{refresh_rate_match.group(1)}Hz"
        else:
            return np.nan

    # Apply the function to fill null values in the 'refresh_rate' column
    df_new['refresh_rate_hz'] = df_new['refresh_rate_hz'].fillna(
        df_new['product_title'].apply(extract_refresh_rate))
    df_new['refresh_rate_hz'] = df_new['refresh_rate_hz'].astype(
        str).str.lower()

    # Extract CPU Speed

    def extract_cpu_speed(title):
        if not title:
            return np.nan
        title = title.lower()

        # Regex pattern to match CPU speed in GHz or MHz
        speed_pattern = r'(\d+(\.\d+)?\s*(ghz|mhz))'

        # Search for CPU speed in the title
        matched_speed = re.search(speed_pattern, title)

        return matched_speed.group(0) if matched_speed else np.nan

    # Apply the function to fill null values in the 'cpu_speed_ghz' column
    df_new['cpu_speed_ghz'] = df_new['cpu_speed_ghz'].fillna(
        df_new['product_title'].apply(extract_cpu_speed))
    df_new['cpu_speed_ghz'] = df_new['cpu_speed_ghz'].astype(str).str.lower()

    # Extract Connectivity Technology

    def extract_connectivity_technology(title):
        if not title:
            return np.nan
        title = title.lower()

        # List of popular connectivity technologies
        connectivity_keywords = ['wi-fi', 'wifi', 'bluetooth',
                                 '5g', '4g', 'lte', 'nfc', 'usb-c', 'usb', 'ethernet']

        # Search for any of the keywords in the title
        matched_tech = [tech for tech in connectivity_keywords if re.search(
            r'\b' + re.escape(tech) + r'\b', title)]

        return matched_tech[0] if matched_tech else np.nan

    # Apply the function to fill null values in the 'connectivity_technology' column
    df_new['connectivity_technology'] = df_new['connectivity_technology'].fillna(
        df_new['product_title'].apply(extract_connectivity_technology))
    df_new['connectivity_technology'] = df_new['connectivity_technology'].astype(
        str).str.lower()

    # Extract CPU Model

    def extract_cpu_model(title):
        if not title:
            return np.nan
        title = title.lower()

        # List of common CPU models
        cpu_keywords = ['snapdragon', 'exynos', 'mediatek', 'apple a15',
                        'a15 bionic', 'dimensity', 'kirin', 'universal chip']

        matched_cpu = [cpu for cpu in cpu_keywords if re.search(
            r'\b' + re.escape(cpu) + r'\b', title)]

        return matched_cpu[0] if matched_cpu else np.nan

    # Apply the function to fill null values in the 'cpu_model' column
    df_new['cpu_model'] = df_new['cpu_model'].fillna(
        df_new['product_title'].apply(extract_cpu_model))
    df_new['cpu_model'] = df_new['cpu_model'].astype(str).str.lower()

    # Extract Color

    def extract_color(title):
        if not title:
            return np.nan
        title = title.lower()

        # List of popular colors
        color_keywords = ['black', 'white', 'blue', 'red', 'silver', 'gold', 'green', 'pink',
                          'purple', 'gray', 'orange', 'purple', 'grey', 'yellow', 'pink', 'bronze', 'midnight']

        matched_color = [color for color in color_keywords if re.search(
            r'\b' + re.escape(color) + r'\b', title)]

        return matched_color[0] if matched_color else np.nan

    # Apply the function to fill null values in the 'color' column
    df_new['color'] = df_new['color'].fillna(
        df_new['product_title'].apply(extract_color))
    df_new['color'] = df_new['color'].astype(str).str.lower()

    # Extract Wireless Carrier

    def extract_wireless_carrier(title):
        if not title:
            return np.nan
        title = title.lower()

        # List of common wireless carriers
        carrier_keywords = ['verizon', 'at&t', 't-mobile', 'sprint',
                            'unlocked', 'cricket', 'boost mobile', 'metro pcs', 'vodafone']

        matched_carrier = [carrier for carrier in carrier_keywords if re.search(
            r'\b' + re.escape(carrier) + r'\b', title)]

        return matched_carrier[0] if matched_carrier else 'unlocked' if 'unlocked' in title else np.nan

    # Apply the function to fill null values in the 'wireless_carrier' column
    df_new['wireless_carrier'] = df_new['wireless_carrier'].fillna(
        df_new['product_title'].apply(extract_wireless_carrier))
    df_new['wireless_carrier'] = df_new['wireless_carrier'].astype(
        str).str.lower()

    # Cellular Technology

    def extract_cellular_technology(title):
        if not title:
            return np.nan
        title = title.lower()

        # List of cellular technologies
        cellular_tech_keywords = ['gsm', 'cdma', '5g', '4g', 'lte', 'dual sim']

        matched_tech = [tech for tech in cellular_tech_keywords if re.search(
            r'\b' + re.escape(tech) + r'\b', title)]

        return matched_tech[0] if matched_tech else np.nan

    # Apply the function to fill null values in the 'cellular_technology' column
    df_new['cellular_technology'] = df_new['cellular_technology'].fillna(
        df_new['product_title'].apply(extract_cellular_technology))
    df_new['cellular_technology'] = df_new['cellular_technology'].astype(
        str).str.lower()

    # Drop rows where both 'Memory Storage Capacity' and 'RAM Memory Installed Size' are NaN
    df_data_filled = df_new.dropna(subset=['ram_gb', 'storage'], how='all')
    # logging.info(f':{df_data_filled.shape}:')

    # After

    # List of columns to check for missing values
    columns_to_check = ['brand', 'model_name', 'product_title', 'price', 'os',
                        'ram_gb', 'storage', 'screen_size_in', 'resolution',
                        'refresh_rate_hz', 'cpu_speed_ghz', 'connectivity_technology',
                        'cpu_model', 'color', 'wireless_carrier', 'cellular_technology']

    # Loop through each column and print the count of missing values
    for column in columns_to_check:
        logging.info(f'{column}: {df_new[column].isna().sum()}')

    logging.info('-' * 50)

    # Clean Data From Unnecessary Chars :

    # Brand

    def clean_brand_names(brands):
        # Dictionary for Japanese to English translations
        translations = {
            'シャープ': 'sharp',
            'ソニー': 'sony',
            '富士通': 'fujitsu',
            'アップル': 'apple',
            'シャオミ': 'xiaomi',
            'エーユー': 'au',
            'グーグル': 'google',
            '京セラ': 'kyocera'
        }

        def clean_single_brand(brand):
            # Remove Japanese text in parentheses
            brand = re.sub(
                r'\([\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uff00-\uff9f]+\)', '', brand)

            # Check if the brand is in Japanese and needs translation
            for jp, en in translations.items():
                if jp in brand:
                    brand = en
                    break

            # Remove any remaining parentheses and clean up whitespace
            brand = re.sub(r'[\(\)]', '', brand).strip()

            return brand.lower()

        cleaned_brands = [clean_single_brand(brand) for brand in brands]
        return cleaned_brands

    brands = list(df_data_filled['brand'])
    df_data_filled.loc[:, 'brand'] = clean_brand_names(brands)

    # Ram

    # Step 1: Drop rows containing "mb" or decimal values
    df_data_filled = df_data_filled[~df_data_filled['ram_gb'].str.contains(
        'mb', case=False, na=False)]  # Remove 'mb'
    df_data_filled = df_data_filled[~df_data_filled['ram_gb'].str.contains(
        r'\b\d+\.\d+\b', na=False)]  # Remove decimal values

    # Step 2: Normalize the 'ram_gb' values
    df_data_filled['ram_gb'] = df_data_filled['ram_gb'].str.replace(
        'gb', '', case=False, regex=False)  # Remove 'gb'
    df_data_filled['ram_gb'] = df_data_filled['ram_gb'].str.strip().str.lower()

    # convert to numeric and coerce errors to NaN (will convert non-numeric values to NaN)
    df_data_filled['ram_gb'] = pd.to_numeric(
        df_data_filled['ram_gb'], errors='coerce')

    # Step 3: Set any value exceeding 24 to 24
    df_data_filled['ram_gb'] = df_data_filled['ram_gb'].where(
        df_data_filled['ram_gb'] <= 24, 16)

    # # Optionally, drop rows with NaN values in 'ram_gb' after conversion
    # df_data_filled = df_data_filled.dropna(subset=['ram_gb'])

    # Convert ram_gb to int after cleaning
    df_data_filled['ram_gb'] = df_data_filled['ram_gb'].astype(int)

    # Storage

    def clean_storage_column(df, storage_col='storage', ram_col='ram_gb'):
        # Create a copy of the input series
        storage_series = df[storage_col].copy()
        ram_series = df[ram_col].copy()

        # First convert everything to lowercase strings and strip whitespace
        storage_series = storage_series.astype(str).str.lower().str.strip()

        def parse_storage(value):
            """Convert storage string to numeric value in GB"""
            try:
                # Skip already invalid values
                if pd.isna(value):
                    return np.nan

                value = str(value).strip()

                # Early return for empty or invalid strings
                if not value or value == 'nan':
                    return np.nan

                # Remove any spaces between numbers and units
                value = re.sub(r'\s+', '', value)

                # Handle different units
                if 'tb' in value:
                    num = float(value.replace('tb', ''))
                    return int(num * 1000)
                elif 'gb' in value:
                    num = float(value.replace('gb', ''))
                    return int(num) if num.is_integer() else np.nan
                elif 'mb' in value:
                    return np.nan
                elif value.replace('.', '').isdigit():
                    num = float(value)
                    return int(num) if num.is_integer() else np.nan

                return np.nan

            except (ValueError, TypeError):
                return np.nan

        # Step 1: Initial conversion to numeric values
        cleaned_values = storage_series.apply(parse_storage)

        # Step 2: Apply business rules in order
        def apply_rules(storage, ram):
            # Return NaN if storage is already NaN
            if pd.isna(storage):
                return np.nan

            storage = int(storage)  # Convert to integer

            # Rule 1: Storage must be >= 16GB
            if storage < 16:
                # Special case: Keep if storage > RAM
                if pd.notna(ram) and storage > ram:
                    return storage
                return np.nan

            # If we get here, the value is valid
            return storage

        # Apply all rules row by row
        result = pd.Series([apply_rules(storage, ram) for storage, ram in zip(
            cleaned_values, ram_series)], index=df.index)

        # Ensure the result is Int64 to handle NaNs correctly
        return result.astype('Int64')

    # Usage example:
    df_data_filled['storage'] = clean_storage_column(df_data_filled)

    # Price

    def scraping_currency(region_cur_name):
        def web_driver():
            options = webdriver.ChromeOptions()
            options.add_argument("--verbose")
            options.add_argument('--no-sandbox')
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument("--window-size=1920,1200")
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--blink-settings=imagesEnabled=false')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disk-cache-size=50000000')
            driver = webdriver.Chrome(options=options)
            return driver

        driver = web_driver()

        try:
            if region_cur_name in ['egp', 'cad', 'jpy', 'aus']:
                link = f"https://www.google.com/search?q=usd+to+{region_cur_name}"
                driver.get(link)

                cur_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'a61j6'))
                )

                if cur_element:
                    conversion_rate = cur_element.get_attribute('value')
                    return float(conversion_rate)  # Ensure it returns a float
                else:
                    logging.error(
                        f"Conversion element not found for {region_cur_name}")
                    return None

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        finally:
            driver.quit()

    def clean_price(price):
        if pd.isna(price):
            return price

        price = str(price)

        # Define a list of unwanted characters and words to remove
        unwanted_chars = [',', '₹', '$', '£', '€', '₣', '¥']
        unwanted_words = ['egp', 'usd', 'inr', 'eur',
                          'gbp', 'aud', 'cny', 'cad', 'jpy', 'aus']

        # Remove currency symbols and unwanted words
        for word in unwanted_words:
            price = re.sub(r'\b' + re.escape(word) + r'\b',
                           '', price, flags=re.IGNORECASE)
        for char in unwanted_chars:
            price = price.replace(char, '')

        price = re.sub(r'[^\d.]', '', price)  # Retain dots for decimal numbers

        if price:
            return round(float(price), 2)
        return None

    def handle_prices(df_data_filled, region_cur_name='usd'):
        prices = df_data_filled['price'].astype(str)
        cleaned_prices = [clean_price(price) for price in prices]
        df_data_filled['price'] = cleaned_prices

        if region_cur_name != 'usd':
            cur = scraping_currency(region_cur_name)

            if cur is not None and isinstance(cur, (float, int)):
                df_data_filled['price'] = df_data_filled['price'] / cur
                df_data_filled['price'] = df_data_filled['price'].round(2)
            else:
                logging.error(
                    "Invalid conversion rate. Cannot perform division.")

        return df_data_filled

    df_data_filled = handle_prices(df_data_filled, region_cur_name)

    # Discount | Screen Size | Refresh Rate | CPU Speed

    def clean_col_val(df, col, remove_patterns=None, strip=True):
        # Remove the provided patterns
        if remove_patterns:
            for pattern in remove_patterns:
                df.loc[:, col] = df[col].str.replace(pattern, "", regex=True)

        # Optionally strip whitespace
        if strip:
            df.loc[:, col] = df[col].str.strip()

        return df

    df_data_filled = clean_col_val(
        df_data_filled, "screen_size_in", remove_patterns=["inches"])
    df_data_filled = clean_col_val(
        df_data_filled, "refresh_rate_hz", remove_patterns=["hz|hertz|ghz"])
    df_data_filled = clean_col_val(
        df_data_filled, "cpu_speed_ghz", remove_patterns=["hz|hertz|ghz"])
    df_data_filled = clean_col_val(
        df_data_filled, "discount", remove_patterns=["-|%"])

    # Wireless Carrier

    df_data_filled['wireless_carrier'] = df_data_filled['wireless_carrier'].map(
        lambda x: 'unlocked' if x == 'unlocked for all carriers' or x == ' unlocked' else x)

    # Add Network col

    # Function to extract the network value

    def extract_network(row):
        # Check if either column is not NaN and contains '4G' or '5G'
        cell_tech = row['cellular_technology'] if pd.notna(
            row['cellular_technology']) else ''
        conn_tech = row['connectivity_technology'] if pd.notna(
            row['connectivity_technology']) else ''

        if '5g' in cell_tech or '5g' in conn_tech:
            return '5g'
        elif '4g' in cell_tech or '4g' in conn_tech:
            return '4g'
        elif '3g' in cell_tech or '3g' in conn_tech:
            return '3g'
        elif '2g' in cell_tech or '2g' in conn_tech:
            return '2g'
        else:
            return None  # or '' if you prefer an empty string

    # Create the new 'network' column
    df_data_filled['network'] = df_data_filled.apply(extract_network, axis=1)

    # Clean Model Name

    class ModelFormatter:
        def __init__(self):
            # Common patterns used across multiple formatters
            self.common_patterns = {
                'storage': r'\b\d+\s*(?:gb|tb)\b',
                'ram_storage': r'\b\d+[-/_]\d+(?:\s*(?:gb|tb))?\b',
                'network': r'\b[45]g\b',
                'colors': r'\b(?:black|white|blue|green|red|purple|gray|grey|gold|silver|orange|yellow|pink|bronze)\b',
                'generic_terms': r'\b(?:smartphone|phone|mobile|device|unlocked|version|new|latest)\b'
            }

            # Compile regex patterns for better performance
            self.compiled_patterns = {
                key: re.compile(pattern, re.IGNORECASE)
                for key, pattern in self.common_patterns.items()
            }

        def _clean_basic(self, text):
            """Basic cleaning operations common to most formatters."""
            if not isinstance(text, str):
                return ""

            text = text.lower().strip()
            text = unicodedata.normalize('NFKD', text).encode(
                'ASCII', 'ignore').decode('utf-8')

            # Remove common patterns
            for pattern in self.compiled_patterns.values():
                text = pattern.sub('', text)

            # Remove hyphens, underscores, and dots
            text = re.sub(r'[-_.]', ' ', text)

            # Remove extra spaces
            return re.sub(r'\s+', ' ', text).strip()

        def format_samsung(self, model_name, title):

            title = self._clean_basic(title)
            model_name = self._clean_basic(model_name)
            # Handle special cases first
            special_patterns = {
                'rugby': r'(rugby\s+\w+\d*)',
                'flip': [(r'z\s*flip\s*(\d+)', True), (r'zflip\s*(\d+)', True), (r'flip\s*(\d+)', False)],
                'fold': [(r'z\s*fold\s*(\d+)', True), (r'zfold\s*(\d+)', True), (r'fold\s*(\d+)', False)]
            }

            # Check rugby pattern
            rugby_match = re.search(special_patterns['rugby'], title)
            if rugby_match:
                return rugby_match.group(1)

            # Check flip/fold patterns
            for device_type, patterns in [('flip', special_patterns['flip']), ('fold', special_patterns['fold'])]:
                for pattern, includes_z in patterns:
                    match = re.search(pattern, title)
                    if match:
                        number = match.group(1)
                        return f"z {device_type}{number}" if includes_z else f"{device_type}{number}"

            # Handle regular models
            model_pattern = r'([a-z]+\s?\d+[a-z]*)'
            suffix_pattern = r'\b(ultra|plus|pro|fe)\b'

            match = re.search(model_pattern, title)
            if match:
                base_model = match.group(1).strip()
                suffix_match = re.search(suffix_pattern, title, re.IGNORECASE)
                return f"{base_model} {suffix_match.group(1)}" if suffix_match else base_model

            return model_name

        def clean_motorola(self, model):

            if not isinstance(model, str):
                return model

            model = self._clean_basic(model)
            model = re.sub(
                r'(?:\b\d+gb\b|\b\d+mb\b|\b\d+gb ram\b|\b\d+gb rom\b|\b\d+mb ram\b).*$', '', model)
            model = re.sub(r'\bmoto\b|\bmotorola\b', '', model)
            model = re.sub(r'\+', '', model)
            model = re.sub(r'\s*\((\d{4})\)', r' \1', model)
            model = re.sub(r'\'(\d{2})', r'\1', model)

            return model.strip()

        def clean_oneplus(self, model):

            if not isinstance(model, str):
                return model

            model = self._clean_basic(model)
            model = re.sub(r'\boneplus\b', '', model)
            model = re.sub(r'\s*\(|\)', ' ', model)

            match = re.search(r'(.*?\b5g\b)', model)
            if match:
                model = match.group(1).strip()
                model = re.sub(r'\b[45]g\b', '', model)

            return model.strip()

        def format_xiaomi(self, model_name):

            cleaned = self._clean_basic(model_name)

            cleaned = re.sub(r'\bxiaomi\b', '', cleaned).strip()
            cleaned = re.sub(r'\bxioami \b', '', cleaned).strip()

            # Remove years (assuming 2015-2025 range)
            cleaned = re.sub(r'\b20[12]\d\b', '', cleaned)

            # Handle POCO models specially
            poco_match = re.search(
                r'\bpoco\s+[a-z]\d+(?:\s+(?:pro|plus|ultra))?\b', cleaned)
            if poco_match:
                return poco_match.group(0)

            cleaned = cleaned.replace('+', ' plus ')
            cleaned = re.sub(r'[-_—–/,]', ' ', cleaned)

            # Clean brand variations
            brand_variations = {
                'xiaomi': 'xiaomi',
                'redmi': 'redmi',
                'poco': 'poco',
                'mi': 'mi'
            }

            for variant, correct in brand_variations.items():
                if cleaned.startswith(variant):
                    cleaned = cleaned.replace(variant, correct, 1)
                    break

            return cleaned.strip()

        def format_honor(self, model):

            model = self._clean_basic(model)
            model = re.sub(r'\bhonor\b', '', model)
            model = re.sub(r'\+', ' plus', model)

            # Handle special keywords
            for keyword in ['pro', 'plus']:
                model = re.sub(fr'\b{keyword}\b', keyword, model)

            return model.strip()

        def format_apple(self, model):

            model = self._clean_basic(model)
            model = re.sub(r'\b(?:apple|iphone)\b', '', model)

            parts = model.split()
            core_model = []

            for part in parts:
                if re.match(r'^\d+', part) or part in ['pro', 'max', 'mini', 'plus', 'x', 'se']:
                    core_model.append(part)

            return ' '.join(core_model).strip()

        def format_oppo(self, model):

            model = self._clean_basic(model)
            model = re.sub(r'\b(oppo)\b', '', model)

            match = re.search(r'(reno\s*\d+\s*\w*|a\d+\s*\w*)',
                              model, re.IGNORECASE)
            if match:
                core_model = match.group(0).strip()
                core_model = re.sub(r'\b[45]g\b', '', core_model)
                return core_model.strip().lower()

            return model.strip()

        def format_nokia(self, model, product_title):

            title = self._clean_basic(product_title)
            title = re.sub(r'^nokia nokia', 'nokia', title)

            patterns = [
                r'nokia\s+(g\d+)',
                r'nokia\s+(x\d+)',
                r'nokia\s+(c\d+(?:\s*plus)?)',
                r'nokia\s+(\d+\.\d+)',
                r'nokia\s+(\d+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, title)
                if match:
                    model = match.group(1)
                    model = model.replace('+', ' plus')
                    model = re.sub(r'\b[45]g\b', '', model)
                    return model.strip()

            return model

        def format_infinix(self, model):

            model = self._clean_basic(model)
            model = re.sub(r'\+', ' plus', model)
            model = re.sub(r'\binfinix\b', '', model)
            model = re.sub(r'\b[45]g\b', '', model)

            return model.strip()

        def format_realme(self, model, title):
            """Enhanced Realme model formatter."""
            title = self._clean_basic(title)
            model = self._clean_basic(model)

            title = re.sub(r'^realme realme', 'realme', title)

            narzo_match = re.search(
                r'realme\s+narzo\s+(\d+i?\s*(?:prime)?)', title)
            if narzo_match:
                return f"narzo {narzo_match.group(1)}"

            model_pattern = r'realme\s+((?:note\s+)?\d+(?:\s*(?:pro|x))?\+?i?|c\s*\d+[a-z]?|gt\s+\d+t?)'
            match = re.search(model_pattern, title)

            if match:
                model = match.group(1).strip()
                model = re.sub(r'c\s+(\d+)', r'c\1', model)
                model = re.sub(r'gt\s+(\d+t?)', r'gt \1', model)
                model = model.replace('+', ' plus')
                return model

            return model

        def format_redmi(self, model):

            model = self._clean_basic(model)
            if not model.startswith('redmi'):
                model = 'redmi ' + model

            model = re.sub(r'\+', ' plus', model)
            model = re.sub(r'(\d+)-(\d+)', r'\1 \2', model)
            model = re.sub(r'\bredmi\b', 'redmi', model)
            model = re.sub(r'\b[45]g\b', '', model)

            return model.strip()

        def clean_ulefone(self, model_name):

            if not model_name:
                return ""

            model = self._clean_basic(model_name)

            series = ['armor', 'power', 'note', 'tiger', 'power armor']
            prefixes = ['ulefone', 'ule-fone', 'ule fone']

            for prefix in prefixes:
                if model.startswith(prefix):
                    model = model[len(prefix):].strip()

            words = model.split()
            cleaned_words = []
            skip_next = False

            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue

                if i < len(words) - 1 and f"{word} {words[i+1]}" == "power armor":
                    cleaned_words.append("Power Armor")
                    skip_next = True
                elif word in series or (word.startswith('x') and len(word) > 1 and word[1:].isdigit()):
                    cleaned_words.append(word)
                elif word.replace('.', '').isdigit() and len(cleaned_words) > 0:
                    cleaned_words.append(word)
                elif word in ['pro', 'max', 'lite', 'ultra']:
                    cleaned_words.append(word)

            return ' '.join(cleaned_words).strip()

        def clean_blackview(self, model_name):

            cleaned = self._clean_basic(model_name)
            cleaned = re.sub(r'\bblackview\b', '',
                             cleaned, flags=re.IGNORECASE)

            model_match = re.search(
                r'\b(?:shark|wave|bv|tab|note|a[0-9]+)\s?\d*[a-zA-Z]*\b', cleaned, re.IGNORECASE)
            if model_match:
                return model_match.group(0).strip()

            return cleaned

        def format_google(self, title):

            # title = self._clean_basic(title)

            pixel_patterns = [
                r'pixel\s*[0-9]{1,2}(?:\s*[a-z]?[a-z]?|\s*\(5g\))?',
                r'pixel\s*[0-9]{1,2}\s*[a-z]?[a-z]?\s*[0-9]{0,3}[gb]?',
                r'pixel\s*[a-z]?[0-9]?\s*[a-z]?[a-z]?\s*[0-9]{0,3}[gb]?',
                r'pixel\s*[0-9]+\s*\+\s*[0-9]+'
            ]

            for pattern in pixel_patterns:
                match = re.search(pattern, title)
                if match:
                    model_name = match.group(0).strip()
                    model_name = re.sub(r'\bpr\b', '', model_name)
                    if 'pro' in title and 'pro' not in model_name:
                        model_name += ' pro'
                    return model_name.strip()

            return title

        def clean_generic(self, model_name, brand_name):

            if not model_name or not isinstance(model_name, str):
                return ""

            cleaned = self._clean_basic(model_name)
            brand_pattern = r'\b' + re.escape(brand_name.lower()) + r'\b'
            cleaned = re.sub(brand_pattern, '', cleaned)

            symbol_replacements = {
                '+': ' plus ',
                '&': ' and ',
                '@': ' at ',
                '%': ' percent ',
                '#': ' number ',
                '/': ' slash ',
                '-': ' dash ',
                '_': ' underscore ',
                '©': '',
                '®': '',
                '™': ''
            }

            for symbol, replacement in symbol_replacements.items():
                cleaned = cleaned.replace(symbol, replacement)

            cleaned = re.sub(r'[^a-z0-9\s-]', ' ', cleaned)
            cleaned = re.sub(r'\'s\b', 's', cleaned)
            cleaned = re.sub(r's\'', 's', cleaned)

            prefixes = ['the ', 'a ', 'an ']
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]

            return cleaned.strip()

        def clean_special_characters(self, model_name):

            if not isinstance(model_name, str):
                return model_name

            # Define color names you want to remove
            colors = r'\b(?:black|white|blue|green|red|purple|gray|grey|gold|silver|orange|titanium|yellow|pink|bronze|midnight|onyx|crystal|gradient|aurora|polar|cosmic)\b'

            words = r'\b(?:android|starry|years|mi|graphite)\b'

            # Replace underscores, dots, and hyphens with an empty string
            model_name = model_name.replace('_', ' ').replace('.', '').replace(
                '-', ' ').replace('+', ' plus ').replace('/', '').replace("'\'", '')

            # Remove color names
            model_name = re.sub(colors, '', model_name, flags=re.IGNORECASE)

            # Remove words
            model_name = re.sub(words, '', model_name, flags=re.IGNORECASE)

            # Remove extra spaces created by the previous replacements
            model_name = re.sub(r'\s+', ' ', model_name).strip()

            return model_name

        def process_model(self, row):
            """Main processing function with enhanced brand routing."""
            brand = row['brand'].lower()
            model_name = row['model_name']
            product_title = row['product_title']

            # Clean special characters from model_name before processing
            model_name = self.clean_special_characters(model_name)

            brand_formatters = {
                'samsung': lambda: self.format_samsung(model_name, product_title),
                'google': lambda: self.format_google(product_title),
                'xiaomi': lambda: self.format_xiaomi(model_name),
                'blackview': lambda: self.clean_blackview(model_name),
                'realme': lambda: self.format_realme(model_name, product_title),
                'ulefone': lambda: self.clean_ulefone(model_name),
                'oneplus': lambda: self.clean_oneplus(model_name),
                'motorola': lambda: self.clean_motorola(model_name),
                'honor': lambda: self.format_honor(model_name),
                'apple': lambda: self.format_apple(model_name),
                'oppo': lambda: self.format_oppo(model_name),
                'nokia': lambda: self.format_nokia(model_name, product_title),
                'redmi': lambda: self.format_redmi(model_name),
                'infinix': lambda: self.format_infinix(model_name),

            }

            if brand in brand_formatters:
                return brand_formatters[brand]()
            else:
                # Remove the brand name from model_name if it's not handled by a specific function
                cleaned_model = self._clean_basic(model_name)

                # Remove the brand name from the model_name if present
                cleaned_model = re.sub(
                    r'\b' + re.escape(brand) + r'\b', '', cleaned_model, flags=re.IGNORECASE).strip()

                return cleaned_model

    formatter = ModelFormatter()
    df_data_filled['model_name'] = df_data_filled.apply(
        formatter.process_model, axis=1)

    # Replace empty strings and spaces with NaNs
    df_data_filled.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Replace all zeroes with NaNs
    df_data_filled.replace(0, np.nan, inplace=True)

    # DataFrame with filled model names
    # df_data_filled = df_data_filled[~df_data_filled['model_name'].isna()]
    df_data_filled = df_data_filled.dropna(axis=0, subset=['model_name'])

    # Convert columns datatypes
    df_data_filled['date_column'] = pd.to_datetime(
        df_data_filled['date_column'], errors='coerce')
    df_data_filled['refresh_rate_hz'] = pd.to_numeric(
        df_data_filled['refresh_rate_hz'], errors='coerce')
    df_data_filled['screen_size_in'] = pd.to_numeric(
        df_data_filled['screen_size_in'], errors='coerce')

    logging.info(df_data_filled.info())

    return df_data_filled.to_json(orient='records')


def main(input: dict) -> dict:

    # A list of product URLs
    region_cur_name = input['region_cur_name']
    file_path = input["file_path"]

    # For testting file locally.
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    try:
        # Use requests to fetch the data from the URL
        response = requests.get(file_path)
        response.raise_for_status()
        data = response.json()

        output_data = data['output']['AmazonData']['scraped_data']['scraped_data']

        processed_data = process_data(output_data, region_cur_name)

        return {
            "processed_data": processed_data
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL: {str(e)}")
        return {"error": str(e)}

    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return {"error": str(e)}
