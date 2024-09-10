from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import json

driver = webdriver.Chrome()

metadata_dict = {}

# Scrape the article links and titles
for i in range(0, 26):
    url = f"https://www.lawspot.gr/nomika-blogs?page={i}"
    driver.get(url)
    
    article_xpath = '//div[contains(@class, "blog-item")]'
    href_xpath = './/a'
    title_xpath = './/div[@class="info p-20"]/h4/text()'
    
    articles = driver.find_elements(By.XPATH, article_xpath)
    
    for article in articles:
        a_tag = article.find_element(By.XPATH, href_xpath)
        href = a_tag.get_attribute('href')
        title = article.find_element(By.XPATH, './/div[@class="info p-20"]/h4').text
        print(f"Href: {href}, Title: {title}")
        breadcrumb = f"Νομική αρθρογραφία > {title}"
        metadata_dict[breadcrumb] = href
# Initialize the new metadata dictionary
new_metadata_dict = {}

# Scrape the content of each article
for idx, (breadcrumb, url) in enumerate(metadata_dict.items(), start=1):
    article_name = f"lawspot{idx}"
    file_name = f"{article_name}.txt"
    driver.get(url)
    
    paragraphs_xpath = '//div[@class="node-body float-left w-100"]//p'
    paragraphs = driver.find_elements(By.XPATH, paragraphs_xpath)
    
    article_text = "\n".join([paragraph.text for paragraph in paragraphs])
    print(article_text)
    
    # Write article content to file
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(article_text)
    
    # Update the new metadata dictionary
    new_metadata_dict[article_name] = breadcrumb

# Save the new metadata dictionary to a JSON file
with open('lawspot_txt.json', 'w', encoding='utf-8') as json_file:
    json.dump(new_metadata_dict, json_file, ensure_ascii=False, indent=4)

driver.quit()
