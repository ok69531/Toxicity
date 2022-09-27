#%%
import os
import re
import sys
import time
import json

import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index

from tqdm import tqdm


#%%
# !pip install chromedriver_autoinstaller
from bs4 import BeautifulSoup
import chromedriver_autoinstaller

import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementNotInteractableException

# from html_table_parser import parser_functions as parser

pd.set_option('mode.chained_assignment', None)


#%%
url = 'https://www.echemportal.org/echemportal/property-search'

option = webdriver.ChromeOptions()
option.add_argument('window-size=1920,1080')

driver_path = chromedriver_autoinstaller.install()
driver = webdriver.Chrome(driver_path, options = option)
driver.implicitly_wait(3)
driver.get(url)


#%%
deselect_path = '//*[@id="datasources-panel-1"]/div/div/div/a[2]'
deselect = driver.find_element_by_xpath(deselect_path)
deselect.click()
time.sleep(0.5)

echa_path = '//*[@id="datasources-panel-1"]/div/echem-search-sources/div/div/div/div[2]/echem-checkbox/div'
echa = driver.find_element_by_xpath(echa_path)
echa.click()
time.sleep(0.5)

query_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[1]/echem-query-builder/div[2]/div/button'
driver.find_element_by_xpath(query_path).click()
time.sleep(0.5)

tox_button_path = '//*[@id="QU.SE.7-toxicological-information-header"]/div/div[1]/button'
driver.find_element_by_xpath(tox_button_path).click()
time.sleep(0.5)

reprod_path = '//*[@id="QU.SE.7.8-toxicity-to-reproduction-header"]/div/div[2]'
driver.find_element_by_xpath(reprod_path).click()
time.sleep(0.5)

reprod_tox_path = '//*[@id="QU.SE.7.8-toxicity-to-reproduction"]/div/div/div[1]/div[3]/button'
driver.find_element_by_xpath(reprod_tox_path).click()
time.sleep(0.5)

info_type_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[1]/div/div/div/ng-select/div/div'
driver.find_element_by_xpath(info_type_path).click()
time.sleep(0.5)

experiment_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[3]'
driver.find_element_by_xpath(experiment_path).click()
time.sleep(0.5)

tg_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[4]/div/div/div/ng-select/div/div'
driver.find_element_by_xpath(tg_path).click()
time.sleep(0.5)

tg421_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[16]'
driver.find_element_by_xpath(tg421_path).click()
time.sleep(0.5)

save_path = '/html/body/echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/ngb-accordion[1]/div[2]/div[2]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/div/button[2]'
driver.find_element_by_xpath(save_path).click()
time.sleep(0.5)

search_path = '/html/body/echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/div[2]/div/button'
driver.find_element_by_xpath(search_path).click()


#%%
result_ = []

page_num_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]'
page_num = int(driver.find_element_by_xpath(page_num_path).text.split(' ')[-1])

start = time.time()
for p in range(1, page_num + 1):
    row_num_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr'
    row_num = len(driver.find_elements_by_xpath(row_num_path))
    
    row = tqdm(range(1, row_num + 1), file = sys.stdout)
    
    for i in row:
        src_dict = {}
        
        chem_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[3]/a'
        property_url = driver.find_element_by_xpath(chem_path % i).get_attribute('href')
        src_dict['link'] = property_url
        
        driver.execute_script('window.open('');')
        driver.switch_to.window(driver.window_handles[1])
        driver.get(property_url)
        
        try:
            accept_path = '/html/body/div[1]/div/div[2]/div[2]/button[1]'
            driver.find_element_by_xpath(accept_path).send_keys(Keys.ENTER)
        except NoSuchElementException:
            pass
        
        try:
            src = driver.page_source
            # soup = BeautifulSoup(src, 'html.parser')
            src_dict['src'] = src
            
            result_.append(src_dict)
        
        except AttributeError:
            pass

        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        
        row.set_postfix({'page': p})
    
    if p < page_num:
        next_page_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]/a[3]'
        driver.find_element_by_xpath(next_page_path).click()
        time.sleep(1.5)
    
    p += 1


print(time.time() - start)


#%%
json.dump(result_, open('tg421_page_src.json', 'w'))
