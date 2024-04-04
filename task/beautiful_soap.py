import time
from selenium import webdriver
from bs4 import BeautifulSoup
import random
from selenium.webdriver.common.by import By
import json

data = dict()

driver = webdriver.Edge(
    service=webdriver.EdgeService(executable_path='/home/nomodeset/下载/edgedriver_linux64/msedgedriver')
)


def get_info():
    prefix = 'https://papers.cool'
    driver.get(prefix)
    html = driver.page_source
    soap = BeautifulSoup(html, 'html.parser')

    for query in soap.select('p.category-item a[target="_blank"]:not([title])'):
        url, title = prefix + query.attrs['href'], query.get_text()
        if '?' not in url:
            yield url, title

    for query in soap.select('p.venue-item a[target="_blank"]:not([title])'):
        url, title = prefix + query.attrs['href'], query.get_text()
        if '?' not in url:
            yield url, title

    driver.close()


def catch_contents(url: str, title: str) -> None:
    print(f'{title} - {url}')
    driver.get(url)

    original_top = 0
    while True:
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight + 200);")
        time.sleep(1 + random.random())  # 停顿一下
        check_height = driver.execute_script(
            "return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
        if check_height == original_top:  # 判断滑动后距顶部的距离与滑动前距顶部的距离
            break
        original_top = check_height

    html = driver.page_source
    soap = BeautifulSoup(html, 'html.parser')

    # filename = f'/home/nomodeset/code/material-lm/workspace/wget-lists/{title}'.replace(' ', '-')

    paper_titles = []
    for query in soap.select('h2.title a.title-link'):
        ss = query.get_text(separator='\n')
        print(ss)
        paper_titles.append(ss)

    paper_summaries = []
    for query in soap.select('.panel.paper .summary'):
        ss = query.get_text(separator='\n')
        print(ss)
        paper_summaries.append(ss)

    paper_prompt_urls = []
    for query in soap.select('h2.title a.title-copy[onclick]'):
        ss = query.attrs['onclick']
        start = 0
        start = ss.find("'", start) + 1
        end = ss.find("'", start)
        if 'arxiv' in url:
            ss = 'https://papers.cool/arxiv/kimi?paper=' + ss[start: end]
        elif 'venue' in url:
            ss = 'https://papers.cool/venue/kimi?paper=' + ss[start: end]
        else:
            raise
        print(ss)
        paper_prompt_urls.append(ss)

    paper_urls = []
    for query in soap.select('h2.title a.title-pdf[onclick]'):
        ss = query.attrs['onclick']
        start = 0
        for _ in range(3):
            start = ss.find("'", start) + 1
        start = ss.find("http", start)
        end = ss.find("'", start)
        ss = ss[start:end]
        print(ss)
        paper_urls.append(ss)

    driver.back()

    data[title] = dict(
        original_url=url,
        paper_titles=paper_titles,
        paper_urls=paper_urls,
        paper_summaries=paper_summaries,
        paper_prompts_urls=paper_prompt_urls
    )


def main():
    for url, title in get_info():
        try:
            catch_contents(url, title)
        except:
            continue
    with open('/home/nomodeset/code/material-lm/workspace/data.json', 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    main()
