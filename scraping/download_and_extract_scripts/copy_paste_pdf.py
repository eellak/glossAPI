import os
import pyperclip
import time
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def main():
    input_folder = "/home/fivos/Projects/GlossAPI/downloaded_texts/ebooks/ebooks"  # Replace with your input folder path
    output_folder = os.path.join(input_folder, "paste_texts")

    # Create 'paste_texts' folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    

    options = Options()
    options.binary_location = "/usr/bin/firefox"
    driver = webdriver.Firefox()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        driver.get(f'file:///{pdf_path}')
        copy_all(driver)
        text = pyperclip.paste()

        # Write the copied text to a file with the same name as the PDF but with a .txt extension
        text_filename = os.path.splitext(pdf_file)[0] + ".txt"
        text_filepath = os.path.join(output_folder, text_filename)

        with open(text_filepath, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        print(f"Text from {pdf_file} saved to {text_filename}")

    driver.quit()

def copy_all(driver):
    wait = WebDriverWait(driver, 10)
    body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    driver.execute_script("arguments[0].scrollIntoView();", body)
    driver.execute_script("arguments[0].focus();", body)
    actions = ActionChains(driver)
    actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()  # Select all
    time.sleep(1)
    actions.key_down(Keys.CONTROL).send_keys('c').key_up(Keys.CONTROL).perform()  # Copy
    time.sleep(3)

def greekify(text):
    try:
        encoded_text = text.encode('latin1', 'replace').decode('iso-8859-7')
    except UnicodeDecodeError:
        print("There was an error decoding the text.")
    else:
        return encoded_text
    return None

if __name__ == "__main__":
    main()
