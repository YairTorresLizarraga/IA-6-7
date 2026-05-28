import time
from urllib.request import urlopen, Request

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from os import path, makedirs

from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from hashlib import sha256

persona = "will smith"
count = 100

persona_dir = f"dataset/{persona}"

already_downloaded: set[str] = set()
skip_elements: set[WebElement] = set()

if not path.isdir(persona_dir):
    makedirs(persona_dir)

driver = webdriver.Firefox()
driver.get(f"https://www.google.com/search?tbm=isch&q={persona}")

while count > 0:
    search_results = WebDriverWait(driver, 60).until(
        expected_conditions.presence_of_element_located((By.ID, "search"))
    )
    images = search_results.find_elements(by=By.TAG_NAME, value="img")
    for image in images:
        if image in skip_elements:
            continue

        skip_elements.add(image)

        try:
            if count == 0:
                break

            alt = image.get_attribute("alt")
            if alt is None or len(alt.strip()) == 0:
                continue

            preview_src = image.get_attribute("src")

            image.click()
            time.sleep(1)

            full_res_image = WebDriverWait(driver, 2).until(
                expected_conditions.presence_of_element_located((By.CSS_SELECTOR, ".sFlh5c.FyHeAf.iPVvYb"))
            )

            src = full_res_image.get_attribute("src")
            extension = src.split(".")[-1].split("?")[0].split("&")[0].split("/")[0].split("%")[0]

            src_hash = sha256(src.encode()).hexdigest()
            dest = f"{persona_dir}/{src_hash}.{extension}"
            if src_hash in already_downloaded or path.exists(dest):
                continue

            req = Request(
                src,
                data=None,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:149.0) Gecko/20100101 Firefox/149.0"
                }
            )
            image_data = urlopen(req).read()
            already_downloaded.add(src_hash)

            if len(image_data) == 0:
                continue
            with open(dest, "wb") as f:
                f.write(image_data)
            print(f"Saved {dest}")

            time.sleep(0.25)
            count -= 1
        except:
            pass

    ActionChains(driver) \
        .scroll_by_amount(0, 1000) \
        .perform()

driver.quit()
