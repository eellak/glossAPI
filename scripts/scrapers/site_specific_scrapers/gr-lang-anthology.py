from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import json
import time
import parsel

def main():
    driver = webdriver.Chrome()
    url = "https://www.greek-language.gr/greekLang/literature/anthologies/new/contents.html"
    metadata_dict = get_links(url, driver)
    for metadata, url in list(metadata_dict.items()):
        new_metadata = get_links(url, driver, 0.5)
        if new_metadata:
            metadata_dict.update(new_metadata)
            del metadata_dict[metadata]
    filename = "anthologies"
    download_articles(metadata_dict, filename, driver)
    driver.quit()

def get_links(url, driver, sleep=3.5):
    driver.get(url)
    time.sleep(sleep)
    try:
        greek = driver.find_element(By.XPATH, "//li[@id='unav_greek']")
        greek.click()
    except:
        print("element not found")

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
        link = f"https://www.greek-language.gr/greekLang/literature/anthologies/new/{href}"
        print(breadcrumb, " : ", link)
        metadata_dict[breadcrumb] = link
    return metadata_dict

def download_articles(metadata_dict, filename, driver):
    new_metadata_dict = {}
    idx = 1
    for (breadcrumb, url) in metadata_dict.items():
        article_name = f"{filename}{idx}"
        file_name = f"{article_name}.txt"
        driver.get(url)
        
        paragraphs = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//p')))
        title, author = "", ""
        try:
            author = driver.find_element(By.XPATH, "//h1[@class='author']").text
            title = driver.find_element(By.XPATH, "//h1[@class='stitle']").text
        except NoSuchElementException:
            pass
        article_text = ""
        for p in paragraphs:
            if p.text == "Η σελίδα που προσπαθήσατε να προσπελάσετε δεν υπάρχει (ακόμη).":
                print(f"{url} is not a valid address")
                continue
            article_text += f"\n{p.text}"
        if article_text.strip() == "":
            print(f"{url} is not a valid address")
            continue
        if title and author:
            article_text = f"{title}\n{author}\n{article_text}"
        else:
            title = breadcrumb.split(">")[-1]
            article_text = title + article_text
        idx += 1
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(article_text)
        new_metadata_dict[article_name] = breadcrumb
    with open(f'{filename}_txt.json', 'w', encoding='utf-8') as json_file:
        json.dump(new_metadata_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()