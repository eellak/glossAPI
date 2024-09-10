from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import time
import parsel

driver = webdriver.Chrome()

url = "https://www.greek-language.gr/greekLang/studies/summary/contents.html"
driver.get(url)
time.sleep(1)
try:
    greek = driver.find_element(By.XPATH, "//li[@id='unav_greek']")
    greek.click()
except:
    print("element not found")

# Get the HTML content of the div with id="colM"
colM_element = driver.find_element(By.ID, "colM")
html_content = colM_element.get_attribute('outerHTML')


selector = parsel.Selector(text=html_content)

links = selector.xpath('//a[@href and not(ancestor::*[@class="vsmall"]) and not(ancestor::*[@class="navigbar"])]')
hrefs_and_texts = [(link.xpath('@href').get(), link.xpath('text()').get()) for link in links]

metadata_dict = {}
for href, text in hrefs_and_texts:
    if not text or text.strip() in ["<", ">"]:
        continue
    breadcrumb = f"greek-language.gr > Περίληψη (2000) > {text}"
    link = f"https://www.greek-language.gr/greekLang/studies/summary/{href}"
    print(breadcrumb, " : ", link)
    metadata_dict[breadcrumb] = link

print(metadata_dict)

new_metadata_dict = {}

# Scrape the content of each article
for idx, (breadcrumb, url) in enumerate(metadata_dict.items(), start=1):
    article_name = f"perilhpsh{idx}"
    file_name = f"{article_name}.txt"
    driver.get(url)
    
    paragraphs_xpath = '//p'
    paragraphs = driver.find_elements(By.XPATH, paragraphs_xpath)
    title = breadcrumb.split(">")[-1]
    article_text = "\n".join([paragraph.text for paragraph in paragraphs])
    article = f"{title}" + article_text
    # Write article content to file
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(article)
    
    # Update the new metadata dictionary
    new_metadata_dict[article_name] = breadcrumb
# Save the new metadata dictionary to a JSON file
with open('perilhpsh_txt.json', 'w', encoding='utf-8') as json_file:
    json.dump(new_metadata_dict, json_file, ensure_ascii=False, indent=4)


driver.quit()
