import json
import os


def main():
    with open('/home/nomodeset/code/material-lm/workspace/data.json', 'r') as file:
        data: dict = json.load(file)
    for title, item in data.items():
        urls = item['paper_urls']
        dir_name = f'/home/nomodeset/code/material-lm/workspace/wget-lists/{title}'.replace(' ', '-')
        os.makedirs(dir_name, exist_ok=True)
        with open(dir_name + '/wget-list.txt', 'w') as file:
            for url in urls:
                print(url)
                file.write(url + '\n')


if __name__ == '__main__':
    main()

