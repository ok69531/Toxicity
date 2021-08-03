#%%
import time
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementNotInteractableException

#%%
url = 'https://www.echemportal.org/echemportal/property-search'

options = webdriver.ChromeOptions()
options.add_argument('window-size=1920,1080')

driver_path = os.getcwd()+'\chromedriver.exe'
driver = webdriver.Chrome(driver_path, options = options)
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


acute_button_path = '//*[@id="QU.SE.7.2-acute-toxicity-header"]/div/div[2]'
acute_button = driver.find_element_by_xpath(acute_button_path)
acute_button.click()


inh_button_path = '//*[@id="QU.SE.7.2-acute-toxicity"]/div/div/div[2]/div[3]/button'
inh_button = driver.find_element_by_xpath(inh_button_path)
inh_button.click()

# ------------------------------------------------------------------------------ #

info_type_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[1]/div/div/div/ng-select/div/span'
info_button = driver.find_element_by_xpath(info_type_path)
info_button.click()


exp_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[3]'
exp_button = driver.find_element_by_xpath(exp_path)
exp_button.click()


guide_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[4]/div/div/div/ng-select/div/span'
guide_button = driver.find_element_by_xpath(guide_path)
guide_button.click()


oecd_gui_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[7]'
oecd_gui_button = driver.find_element_by_xpath(oecd_gui_path)
oecd_gui_button.click()

# ------------------------------------------------------------------------------ #

save_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/div/button[2]'
save_button = driver.find_element_by_xpath(save_path)
save_button.click()

# ------------------------------------------------------------------------------ #

search_path = '//*[@id="top"]/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/div/div/button'
search_button = driver.find_element_by_xpath(search_path)
search_button.click()



#%%
df = pd.DataFrame(columns = {'Chem_Name', 'CasRN', 'Descriptor', 'Value'})
df.columns = ['Chemical_Name'] + ['CasRN'] + ['Value'] + ['Descriptor']

page_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]'
page_length_tmp = driver.find_element_by_xpath(page_path).text
page_length = int(page_length_tmp.split(' ')[-1])

p = 69

start = time.time()

while p <= page_length:
    
    chem_num_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr'
    chem_num = len(driver.find_elements_by_xpath(chem_num_path)) 
    
    row_length = tqdm(range(1, chem_num + 1), file=sys.stdout)
    i=5
    for i in row_length:
        row_data_tmp = []
        panel_group_tmp = []
        
        path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]' % i 
    
        chem_name_path = path + '/td[1]/a[1]'
        row_data_tmp.append(driver.find_element_by_xpath(chem_name_path).text)
        
        casrn_path = path + '/td[2]'
        row_data_tmp.append(driver.find_element_by_xpath(casrn_path).text.split('\n')[0])
        
        echa_path = path + '/td[3]/a'
        echa_url_path = driver.find_element_by_xpath(echa_path)
        echa_url = echa_url_path.get_attribute('href')

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
                inh_path = '//*[@id="SubNav7_3_3"]'
                inh_url = driver.find_element_by_xpath(inh_path)
                inh_url.click()

                try:
                    ''' Result and discussion이 sBlock에 속해있는 경우 '''
                    ld_path = driver.find_elements_by_xpath('//div[@class="sBlock"]')[-1]
                    ld_tmp = ld_path.text.split('\n')
                    
                    if (ld_tmp[0] == 'Effect levels'):
                        row_data_tmp.append(ld_tmp[ld_tmp.index('Dose descriptor:') + 3])
                        row_data_tmp.append(ld_tmp[ld_tmp.index('Dose descriptor:') + 1])
                    
                    else:
                        ''' Result and discussion이 panel-group에 속해있는 경우 '''
                        try:
                            panel_df = pd.DataFrame(columns = {'Chem_Name', 'CasRN', 'Descriptor', 'Value'})
                            panel_df.columns = ['Chemical_Name'] + ['CasRN'] + ['Value'] + ['Descriptor']
                            
                            panel_len_path = driver.find_elements_by_xpath('//div[@class="panel-group"]')
                            group_len = len(panel_len_path)
                            panel_len = len(panel_len_path[-1].text.split('\n'))
                        
                             
                            for j in range(1, panel_len + 1):
                                effect_button_path = '//*[@id="headingEffectLevels%dCollapseGroup%d"]' % (j, group_len)
                                effect_button = driver.find_element_by_xpath(effect_button_path)
                                time.sleep(0.2)
                                effect_button.click()
                                time.sleep(0.2)
                            
                            for j in range(1, panel_len + 1):
                                panel_group_tmp = row_data_tmp.copy()
                            
                                effect_path = '//*[@id="collapseEffectLevels%dCollapseGroup%d"]' % (j, group_len)
                                effect_tmp = driver.find_element_by_xpath(effect_path)
                                time.sleep(0.5)
                                effect = effect_tmp.text.split('\n')
                                
                                panel_group_tmp.append(effect[effect.index('Dose descriptor:') + 3])
                                panel_group_tmp.append(effect[effect.index('Dose descriptor:') + 1])
                                
                                panel_df = panel_df.append(pd.Series(panel_group_tmp, index = panel_df.columns),
                                                        ignore_index = True)
                                
                        except IndexError:
                            row_data_tmp.append(None)
                            row_data_tmp.append(None)
                            
                        except NoSuchElementException:
                            row_data_tmp.append(None)
                            row_data_tmp.append(None)
                        
                        except ElementNotInteractableException:
                            row_data_tmp.append(None)
                            row_data_tmp.append(None)
                    
                except ValueError:
                    row_data_tmp.append(None)
                    row_data_tmp.append(None)
                
                except IndexError:
                    row_data_tmp.append(None)
                    row_data_tmp.append(None)
                
                except NoSuchElementException:
                    row_data_tmp.append(None)
                    row_data_tmp.append(None)
                    
            except StaleElementReferenceException:
                row_data_tmp.append(None)
                row_data_tmp.append(None)
                
        except ElementNotInteractableException:
            row_data_tmp.append(None)
            row_data_tmp.append(None)  
            
        except NoSuchElementException:
            row_data_tmp.append(None)
            row_data_tmp.append(None)
        
        
        # df = df.append(pd.Series(row_data_tmp, index = df.columns), ignore_index = True)
        if (len(row_data_tmp) == 4):
            df = df.append(pd.Series(row_data_tmp, index = df.columns), ignore_index = True)
        else:
            df = df.append(panel_df, ignore_index = True)
        
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        
        row_length.set_postfix({'page': p})
        # row_length.set_description('page = %d' % p)
    
    next_page_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]/a[3]'
    driver.find_element_by_xpath(next_page_path).click()
    
    p = p+1

print('\ntime:', time.time() - start)


#%%
df.to_excel('echa_inhalation.xlsx', header = True, index = False)
