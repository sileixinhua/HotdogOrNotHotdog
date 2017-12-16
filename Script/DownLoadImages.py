# -*- coding:utf-8 -*-  
# coder:橘子派_司磊
# 2017年10月30日 23点58分
# 下载百度图片里的图片，搜索关键字为热狗
import os
import re
import urllib
import json
import socket
import urllib.request
import urllib.parse
import urllib.error
import time

# 设置延迟时间
timeout = 5
socket.setdefaulttimeout(timeout)

class Crawler:
    __time_sleep = 0.1
    __amount = 0
    __start_amount = 0
    __counter = 0
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}

    def __init__(self, t=0.1):
        self.time_sleep = t

    def __save_image(self, rsp_data, word):

        if not os.path.exists("./" + word):
            os.mkdir("./" + word)

        self.__counter = len(os.listdir('./' + word)) + 1
        for image_info in rsp_data['imgs']:
            try:
                time.sleep(self.time_sleep)
                fix = self.__get_suffix(image_info['objURL'])
                urllib.request.urlretrieve(image_info['objURL'], './' + word + '/' + str(self.__counter) + str(fix))
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print("已有" + str(self.__counter) + "张图")
                self.__counter += 1
        return

    @staticmethod
    def __get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    @staticmethod
    def __get_prefix(name):
        return name[:name.find('.')]

    def __get_images(self, word='火锅'):
        search = urllib.parse.quote(word)
        pn = self.__start_amount
        while pn < self.__amount:

            url = 'http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=' + search + '&cg=girl&pn=' + str(
                pn) + '&rn=60&itg=0&z=0&fr=&width=&height=&lm=-1&ic=0&s=0&st=-1&gsm=1e0000001e'
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                rsp = page.read().decode('unicode_escape')
            except UnicodeDecodeError as e:
                print(e)
                print('-----UnicodeDecodeErrorurl:', url)
            except urllib.error.URLError as e:
                print(e)
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print(e)
                print("-----socket timout:", url)
            else:
                rsp_data = json.loads(rsp)
                self.__save_image(rsp_data, word)
                print("下载下一页")
                pn += 60
            finally:
                page.close()
        print("下载任务结束")
        return

    def start(self, word, spider_page_num=1, start_page=1):
        self.__start_amount = (start_page - 1) * 60
        self.__amount = spider_page_num * 60 + self.__start_amount
        self.__get_images(word)

if __name__ == '__main__':
    crawler = Crawler(0.05)
    crawler.start('火锅', 5000, 1)