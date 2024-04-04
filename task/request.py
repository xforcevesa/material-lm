import requests
import beautiful_soap
import json
import time
import random
import os
import threading

os.environ['http_proxy'] = 'http://127.0.0.1:8889'
os.environ['https_proxy'] = 'http://127.0.0.1:8889'

config_index = 0


def req(url: str, referer: str):
    global config_index
    config = [
        dict(
            cookies={
                "_ga": "GA1.1.1030307598.1711803921",
                "_ga_214H31WLDF": "GS1.1.1712202018.9.1.1712202020.0.0.0",
                "client_id": "\"!731c9ow/kL0FCbDgYP0ojQ"
                             "==?gAWVKwAAAAAAAACMCWNsaWVudF9pZJSMGTExMDg3Mi0xNzEyMjAyMDE3LjI1NDUwNzOUhpQu\"",
                "Hm_lpvt_606b976365dabacb1f69823d8de064ee": "1712202020",
                "Hm_lvt_606b976365dabacb1f69823d8de064ee": "1711803922,1711809012,1712151531,1712202019"
            },
            headers={
                "Host": "papers.cool",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": referer,
                "Origin": "https://papers.cool",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "same-origin",
                "Content-Length": "0",
                "TE": "trailers",
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache"
            }
        ),
        dict(
            cookies={
                "_ga": "GA1.1.468400335.1712211076",
                "_ga_214H31WLDF": "GS1.1.1712211075.1.1.1712211173.0.0.0",
                "client_id": "\"!IaAVnujij/MTMRNO/h6ybg"
                             "==?gAWVKwAAAAAAAACMCWNsaWVudF9pZJSMGTExMzI5Mi0xNzEyMjExMDcxLjMwNzMxNDSUhpQu\"",
                "Hm_lpvt_606b976365dabacb1f69823d8de064ee": "1712211174",
                "Hm_lvt_606b976365dabacb1f69823d8de064ee": "1712211075"
            },
            headers={
                "Host": "papers.cool",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": referer,
                "Origin": "https://papers.cool",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "same-origin",
                "Content-Length": "0",
                "TE": "trailers",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
                "Pragma": "no-cache",
                "Cache-Control": "no-cache"
            }
        )
    ]
    config_index = (config_index + 1) % len(config)

    is_stopped = False
    is_stopped_lock = threading.Lock()

    def progress_thread_func():
        while True:
            req_pro = requests.get(
                url=url.replace('kimi', 'progress'),
                **config[config_index]
            )
            if req_pro.status_code == 200:
                progress = float(req_pro.text)
                if progress < 0:
                    print(f'Pending request: {-progress:.2f} requests')
                else:
                    print(f'Loading progress: {progress * 100:.2f}%')
            with is_stopped_lock:
                if is_stopped:
                    break
            time.sleep(0.5)

    progress_thread = threading.Thread(target=progress_thread_func)

    progress_thread.start()

    request = requests.post(
        url=url,
        **config[config_index]
    )

    with is_stopped_lock:
        is_stopped = True

    progress_thread.join()

    soap = beautiful_soap.BeautifulSoup(request.text, 'html.parser')
    ss = soap.get_text()
    ss = [s for s in ss.split('\n') if len(s)]
    res = []
    for index, s in enumerate(ss):
        if s[0] == 'Q':
            res.append([s[2:].strip(), index])
        if s[0] == 'A':
            ss[index] = s[2:].strip()
    rr = []
    for index in range(len(res)):
        start_index = res[index][1] + 1
        end_index = res[index + 1][1] if len(res) > index + 1 else len(ss)
        rr.append({
            'question': res[index][0],
            'answer': '\n'.join(ss[start_index: end_index])
        })
    return rr


def example():
    rr = req(
        url="https://papers.cool/arxiv/kimi?paper=2404.02426",
        referer="https://papers.cool/arxiv/cs.AI"
    )
    for r in rr:
        print('question: ', r['question'])
        print('answer: ', r['answer'])


def main():
    with open('/home/nomodeset/code/material-lm/workspace/data.json', 'r') as file:
        data: dict = json.load(file)
    for title, item in data.items():
        referer = item['original_url']
        print(referer)
        for url in item['paper_prompts_urls']:
            while True:
                try:
                    print(f'{url = },{referer = }')
                    rr = req(url=url, referer=referer)
                    print(rr)
                    data[title]['question_answers'] = rr
                    time.sleep(random.random() * 5 + 5)
                    break
                except:
                    continue
    with open('/home/nomodeset/code/material-lm/workspace/data-prompts.json', 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    main()
