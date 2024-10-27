# Amazon Scraper Data Transformations

## Project Overview

This repository contains data transformation scripts for product information scraped from Amazon. The data is collected using a custom-built scraper leveraging `aiohttp` and `BeautifulSoup`, and includes product details such as price, discount, rating, and reviews. This repository focuses on cleaning, transforming, and formatting the scraped data to make it more suitable for analysis and further processing.

## Features

- **Data Cleaning**: Standardizes product names, removes unwanted characters, and formats price values.
- **Price Formatting**: Cleans up price fields by removing commas, currency symbols, and other formatting characters to ensure consistency.
- **Model Name Extraction**: Extracts and formats the model names for different brands (e.g., Oppo, Nokia, Realme, HONOR) using specific rules for each brand.
- **RAM & Storage Capacity Parsing**: Parses and assigns values for RAM and storage capacity based on keywords (e.g., 'GB') present in the product titles.
- **Category Filtering**: Excludes non-phone products by verifying product details rather than relying solely on category labels.

## Data Sources

The data used in this project is scraped from Amazon's product pages, focusing on categories related to mobile phones. The scraper retrieves multiple product details from each page, such as:

- **Product Title**
- **Brand Name**
- **Price**
- **Discounts**
- **Ratings & Reviews**
- **RAM and Storage Capacity**

## Data Transformation Process

### 1. **Price Cleaning and Currency Standardization**

To ensure consistent pricing across regions, the transformation pipeline includes currency standardization. Prices are unified by removing commas, decimal points, and currency symbols (e.g., 'EGP') and then converting all amounts to USD as the base currency. For example:

- `egp37,750.00` → `converted to USD`
- `usd1,200.00 ` → `1200 in unified format`

### 2. **Model Name Extraction**

Custom logic is applied for various phone brands to extract model names in a consistent format:

- **Oppo**: Extracts core model names and handles cases like `'Oppo Reno 12F' → 'Reno 12F'`, while ensuring descriptive terms like '5G' are retained.
- **Nokia**: Strips out the 'Nokia' prefix and extracts the model number, such as `'nokia g60 5g smartphone' → 'g60 5g'`.
- **Realme**: Handles complex cases such as `'12 pro+ 5g' → '12 pro plus 5g'`.

### 3. **RAM & Storage Extraction**

The transformation detects 'GB' or 'gb' values in product titles to assign:

- **RAM Capacity**: The first 'GB' reference typically corresponds to RAM (e.g., 4GB RAM).
- **Storage Capacity**: The second 'GB' reference is assigned to the storage (e.g., 128GB Storage).

### 4. **Non-Phone Filtering**

The transformation process includes checks to remove non-phone products that might have been scraped under incorrect categories. This ensures only mobile phone data is processed further.

## Future Enhancements

1. **Add automated data validation checks**: Implement automated checks to ensure data quality during the transformation process.
2. **Integrate the transformation pipeline with a cloud-based workflow**: Move the pipeline to a cloud environment for more scalable processing.
3. **Extend support for more product categories and brands**: Broaden the scope to include additional product categories beyond mobile phones.
