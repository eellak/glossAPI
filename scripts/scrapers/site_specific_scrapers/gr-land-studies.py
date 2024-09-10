from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time
import parsel

def main():
    driver = webdriver.Chrome()
    visit = ['summary', 'guide', 'discourse', 'europe']
    for site in visit[-1:]:
        url = f"https://www.greek-language.gr/greekLang/studies/{site}/contents.html"
        if site == 'anthologies':
            url = "https://www.greek-language.gr/greekLang/literature/anthologies/new/contents.html"
        metadata_dict = get_links(url, driver, site)
        print(metadata_dict)
        filename = ''
        if site == "summary": filename = "perilhpsh"
        elif site == "guide": filename = "enc_odygos"
        elif site == "discourse": filename = "logos_keimeno"
        elif site == "europe": filename = "gr_in_eu"
        else: continue
        download_articles(metadata_dict, filename, driver)
    driver.quit()

def get_links(url, driver, site):
    driver.get(url)
    time.sleep(3.5)
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
        header = selector.xpath('//div[@class="heading"]/h1[@class="title"]')
        title = header.xpath('text()').get().replace("\xa0", "").strip()
        breadcrumb = f"greek-language.gr > {title} > {text}"
        link = f"https://www.greek-language.gr/greekLang/studies/{site}/{href}"
        print(breadcrumb, " : ", link)
        metadata_dict[breadcrumb] = link
    return metadata_dict

def download_articles(metadata_dict, filename, driver):
    new_metadata_dict = {}
    for idx, (breadcrumb, url) in enumerate(metadata_dict.items(), start=1):
        article_name = f"{filename}{idx}"
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
    with open(f'{filename}_txt.json', 'w', encoding='utf-8') as json_file:
        json.dump(new_metadata_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()