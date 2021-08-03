#%%
# import requests
# from bs4 import BeautifulSoup

#
# import urllib.request
import time
import os
# import random
import numpy as np
import pandas as pd

from tqdm import tqdm
# from bs4 import BeautifulSoup
# from urllib.request import Request
# from urllib.request import urlopen

#
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException


#%%
url = 'https://www.echemportal.org/echemportal/property-search'
html = requests.get(url)

html.text
html.headers
html.status_code
html.ok


#%%
url = 'https://www.echemportal.org/echemportal/property-search'

# options = webdriver.ChromeOptions()
# options.add_argument('window-size=1920,1080')

driver_path = os.getcwd()+'\chromedriver.exe'
driver = webdriver.Chrome(driver_path)
# driver = webdriver.Chrome(driver_path, options = options)
driver.implicitly_wait(2)
driver.get(url)



#%%
des_path = '//*[@id="datasources-panel-1"]/div/div/div/a[2]'
deselect = driver.find_element_by_xpath(des_path)
deselect.click()


echa_path = '//*[@id="datasources-panel-1"]/div/echem-search-sources/div/div/div/div[2]/echem-checkbox'
echa = driver.find_element_by_xpath(echa_path)
echa.click()

# ------------------------------------------------------------------------------ #

query_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[1]/echem-query-builder/div[2]/div/button'
query = driver.find_element_by_xpath(query_path)
query.click()


tox_button_path = '//*[@id="QU.SE.7-toxicological-information-header"]/div/div[1]/button'
tox_button = driver.find_element_by_xpath(tox_button_path)
tox_button.click()


car_button_path = '//*[@id="QU.SE.7-toxicological-information"]/div/div/div/div[2]/button'
car_button = driver.find_element_by_xpath(car_button_path)
car_button.click()

# ------------------------------------------------------------------------------ #

info_type_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[1]/div/div/div/ng-select/div/span'
info_button = driver.find_element_by_xpath(info_type_path)
info_button.click()

# ------------------------------------------------------------------------------ #


select_path = '//*[@id="' + driver.find_element_by_xpath('/html/body/ng-dropdown-panel').get_attribute('id') + '"]/div[1]/button[1]'
select_button = driver.find_element_by_xpath(select_path)
select_button.click()

# ------------------------------------------------------------------------------ #

save_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/div/button[2]'
save_button = driver.find_element_by_xpath(save_path)
save_button.click()

# ------------------------------------------------------------------------------ #

search_path = '//*[@id="top"]/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/div/div/button'
search_button = driver.find_element_by_xpath(search_path)
search_button.click()



#%%
# df = pd.DataFrame(columns = {'Chem_Name', 'CasRN', 'Descriptor', 'Value'})
# df.columns = list(df.columns[2:]) + list(df.columns[:2])

# page_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]'
# page_length_tmp = driver.find_element_by_xpath(page_path).text
# page_length = int(page_length_tmp.split(' ')[-1])

p = 427
start = time.time()
while p <= 2000:
    
    chem_num_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr'
    chem_num = len(driver.find_elements_by_xpath(chem_num_path)) 
    
    row_length = tqdm(range(1, chem_num + 1))
    
    for i in row_length:
        row_data_tmp = []
        
        path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]' % i 
    
        chem_name_path = path + '/td[1]/a[1]'
        row_data_tmp.append(driver.find_element_by_xpath(chem_name_path).text)
        # Chemical_Name.append(driver.find_element_by_xpath(chem_name_path).text)
        
        casrn_path = path + '/td[2]'
        row_data_tmp.append(driver.find_element_by_xpath(casrn_path).text.split('\n')[0])
        # CasRN.append(driver.find_element_by_xpath(casrn_path).text.split('\n')[0])
        
        echa_path = path + '/td[3]/a'
        echa_url_path = driver.find_element_by_xpath(echa_path)
        echa_url = echa_url_path.get_attribute('href')

        # driver.execute_script("window.open('');")
        # driver.switch_to.window(driver.window_handles[1])
        # driver.get(echa_url)
        
        try: 
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(echa_url)
            
            try:
                accept_path = '//*[@id="_viewsubstances_WAR_echarevsubstanceportlet_acceptDisclaimerButton"]'
                accept = driver.find_element_by_xpath(accept_path)
                accept.click()
            
            except NoSuchElementException:
                pass
            
            tox_path = '//*[@id="MainNav7"]/a'
            tox_button = driver.find_element_by_xpath(tox_path)
            tox_button.click()
            
            acute_path = '//*[@id="SubNav7_3"]'
            acute_button = driver.find_element_by_xpath(acute_path)
            acute_button.click()
            
            try:
                endpoint_path = '//*[@id="SubNav7_3_1"]/a'
                endpoint_url = driver.find_element_by_xpath(endpoint_path)
                endpoint_url.click()
        
                try:
                    ld_path = '//*[@id="SectionContent"]/dl[%d]' % 2
                    ld_tmp = driver.find_element_by_xpath(ld_path)
                    ld = ld_tmp.text.split('\n')
                    
                    if ((2 < len(ld)) and (len(ld) <= 12)):
                        row_data_tmp.append(ld[ld.index('Value:') + 1])
                        row_data_tmp.append(ld[ld.index('Dose descriptor:') + 1])
                    
                    elif len(ld) > 13:
                        ld_path = '//*[@id="SectionContent"]/dl[%d]' % 3
                        ld_tmp = driver.find_element_by_xpath(ld_path)
                        ld = ld_tmp.text.split('\n')
                        
                        if ((len(ld) % 2 == 0) and ((2 < len(ld)) and (len(ld) <= 12))):
                            row_data_tmp.append(ld[ld.index('Dose descriptor:') + 3])
                            row_data_tmp.append(ld[ld.index('Dose descriptor:') + 1])
                        else: 
                            row_data_tmp.append(ld[ld.index('Dose descriptor:') + 2])
                            row_data_tmp.append(ld[ld.index('Dose descriptor:') + 1])
                        
                    else:
                        ld_path = '//*[@id="SectionContent"]/dl[%d]' % 1
                        ld_tmp = driver.find_element_by_xpath(ld_path)
                        ld = ld_tmp.text.split('\n')
                        
                        try:
                            row_data_tmp.append(ld[ld.index('Value:') + 1])
                            row_data_tmp.append(ld[ld.index('Dose descriptor:') + 1])
                        except ValueError:
                            row_data_tmp.append(None)
                            row_data_tmp.append(None)
                        
                except ValueError:
                    row_data_tmp.append(None)
                    row_data_tmp.append(None)
                
                except NoSuchElementException:
                    row_data_tmp.append(None)
                    row_data_tmp.append(None)
                    
            
            except StaleElementReferenceException:
                row_data_tmp.append(None)
                row_data_tmp.append(None)
                
        
        except NoSuchElementException:
            row_data_tmp.append(None)
            row_data_tmp.append(None)
            
        df = df.append(pd.Series(row_data_tmp, index = df.columns), ignore_index = True)
        
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        
        row_length.set_postfix({'page': p})
        # tqdm.write('page = ' + str(p))
        
    next_page_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]/a[3]'
    driver.find_element_by_xpath(next_page_path).click()
    
    p = p+1

print('\ntime: ', time.time() - start)



#%%
df_stop = df.copy()
df.to_csv('ld50_ex.csv', header = True, index = False, sep = ',')
df.columns = ['Chemical_Name'] + ['CasRN'] + ['Descriptor'] + ['Value']
df = df[['Chemical_Name', 'CasRN', 'Value', 'Descriptor']]

a = df.iloc[:1146, ].copy()
a = a[['Chemical_Name', 'CasRN', 'Value', 'Descriptor']]
b = df.iloc[1146:, ].copy()
b.columns = ['Chemical_Name'] + ['CasRN'] + ['Value'] + ['Descriptor']

df = pd.concat([a, b], ignore_index = True)


#%%
a = driver.find_element_by_tag_name('echem-checkbox')
b = a.find_element_by_tag_name('div')
c = b.find_element_by_tag_name('input')
id1 = c.get_attribute('id')
id1[-3:-1]
c.get_attribute('class')

driver.find_element_by_id(id1).click()
echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/div[1]
driver.find_element_by_xpath('/html/body/echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance/sesarch/form/ngb-accordion/div/')
# search_box.send_keys('')

# path = '//*[@id="datasources-panel-1"]/div/echem-search-sources/div/div/div '
# path = '//*[@id="datasources-panel-1"]/div/echem-search-sources/div/div/div/div/echem-checkbox'
# path = '//*[@id="4sbvrzga0skrc4c6ym"]'
# a = driver.find_element_by_xpath(path)
# id = a.find_element_by_tag_name('input').get_attribute('id')
# driver.find_element_by_id(id).click()


#%%
# url = 'https://www.echemportal.org/echemportal/property-search'
# req = Request(url, headers = {'User-Agent':'Mozila/5.0'})
# webpage = urlopen(req)
# soup = BeautifulSoup(webpage)

# obj = soup.find('body')


#%%
# 기본 path: '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[2]' % 1
# chemical name은 뒤에 a[1] 추가
# chemical name은 tr을 바꿔가면서

chem_name = []
for i in range(1, 5):
    path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[1]/a[1]' % i
    chem_name_tmp = driver.find_element_by_xpath(path).text
    chem_name.append(chem_name_tmp)

# chemical의 CasRN과 LD50, value는 td를 바꿔가면서
path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[2]' % 1

CasRN = []
for i in range(1, 5):
    path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[2]' % i
    CasRN_tmp = driver.find_element_by_xpath(path).text.split('\n')[0]
    CasRN.append(CasRN_tmp)


# LD50
# LD50 저장되어 있는 링크로 이동
echa_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[3]/a' % 1
echa_url_path = driver.find_element_by_xpath(echa_path)
echa_url = echa_url_path.get_attribute('href')

driver.execute_script("window.open('');")
driver.switch_to.window(driver.window_handles[1])
driver.get(echa_url)


# if accept then yes else pass
accept_path = '//*[@id="_viewsubstances_WAR_echarevsubstanceportlet_acceptDisclaimerButton"]'
accept = driver.find_element_by_xpath(accept_path)
accept.click()


# acute toxicity
acute_path = '//*[@id="SubNav7_3"]'
acute_button = driver.find_element_by_xpath(acute_path)
acute_button.click()


# endpoint summary path
endpoint_path = '//*[@id="SubNav7_3_1"]/a'
endpoint_url = driver.find_element_by_xpath(endpoint_path)
endpoint_url.click()

# endpoint_path = '//*[@id="SubNav7_3_1"]/a'
# endpoint_url_path = driver.find_element_by_xpath(endpoint_path)
# endpoint_url = endpoint_url_path.get_attribute('href')


# driver.execute_script("window.open('');")
# driver.switch_to.window(driver.window_handles[2])
# driver.get(endpoint_url)


# descriptor & LD50 value
ld_path = '//*[@id="SectionContent"]/dl[2]'
descriptor = driver.find_element_by_xpath(ld_path + '/dd[%d]' % 1).text
value = driver.find_element_by_xpath(ld_path + '/dd[%d]' % 2).text


driver.close()


# driver.switch_to.window(driver.window_handles[0])

''' 여기까지 하고 두 번째 chemical로 넘어가게 코드 짜기 (반복문) '''
''' 페이지 넘어갈때 마다 어떻게 바뀌는지도 확인하기 ''''
''' 한 페이지 끝나면 next button 눌러서 페이지 바뀌게 하기? '''
''' page 어떻게 다룰 것인지 ,, '''
''' chemical, CasRN, LD50 dataframe으로 저장 '''

''' 한 페이지 내에 chemical은 5개로 동일한것 같지만 그래도 밑에같은 조건 써서 사용하자 ..! '''
p = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr'
a = driver.find_elements_by_xpath(p)
len(a)
