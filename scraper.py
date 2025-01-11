import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

def scrape_all_biases(main_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(main_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve the main page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    data = []

    # Debug: print all links for inspection
    bias_links = soup.find_all('a', href=True)
    print(f"Total links found: {len(bias_links)}")

    # Locate bias-specific links
    bias_urls = [urljoin(main_url, link['href']) for link in bias_links if '/media-bias-types/?biases=' in link['href']]
    print(f"Total bias links found: {len(bias_urls)}")

    for bias_url in bias_urls:
        print(f"Scraping: {bias_url}")
        bias_response = requests.get(bias_url, headers=headers)

        if bias_response.status_code != 200:
            print(f"Failed to retrieve {bias_url}. Status code: {bias_response.status_code}")
            continue

        bias_soup = BeautifulSoup(bias_response.text, 'html.parser')
        bias_data = {}

        # Extract Bias Title
        title_tag = bias_soup.find('h1')
        bias_data['title'] = title_tag.get_text(strip=True) if title_tag else 'No Title'

        # Extract \"Also known as\"
        also_known_as_section = bias_soup.find(lambda tag: tag.name == 'div' and 'Also known as' in tag.text)
        bias_data['also_known_as'] = also_known_as_section.find_next('div').get_text(strip=True) if also_known_as_section else 'N/A'

        # Extract Description
        description_section = bias_soup.find(lambda tag: tag.name == 'strong' and 'Description' in tag.text)
        bias_data['description'] = description_section.find_next('p').get_text(strip=True) if description_section else 'N/A'

        # Extract Variants and Examples
        variants_section = bias_soup.find(lambda tag: tag.name == 'strong' and 'Variants and Examples' in tag.text)
        variants = []
        if variants_section:
            variant_blocks = variants_section.find_all_next('div', class_='elementor-widget-container')
            for block in variant_blocks:
                title = block.find('strong')
                example = block.find('p')
                if title and example:
                    variants.append({
                        'title': title.get_text(strip=True),
                        'example': example.get_text(strip=True)
                    })
        bias_data['variants_and_examples'] = variants if variants else 'N/A'

        data.append(bias_data)
        time.sleep(1)

    with open('all_bias_types.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("All bias types data has been saved to all_bias_types.json")

# Example usage
scrape_all_biases('https://biaschecker.ai/media-bias-types/')
