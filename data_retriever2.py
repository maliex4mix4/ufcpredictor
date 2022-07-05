import scrapy, string, json, os
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from scrapy.http import HtmlResponse, Request
from scrapy_selenium import SeleniumRequest
import json

list_of_fighters = []
letters = list(string.ascii_lowercase)
real_list = []

for x in letters:
    fi = json.load(open(f"./Data/Fightme/fighters-{x}.json", "r"))
    list_of_fighters.append(fi)

for x in list_of_fighters:
    for i in x:
        real_list.append(i['link'])

# print(real_list)

class UfcscraperSpider(scrapy.Spider):
	name = 'ufcscraper'

	start_urls = real_list

	def parse(self, response):
		results = []
		
		res = {}
		res["name"] = response.css('span.b-content__title-highlight ::text').get().strip()
		res["nname"] = response.css('p.b-content__Nickname ::text').get().strip()
		res['win'] = int(response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[0].strip())
		res['lose'] = int(response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[1].strip())

		if "NC" in response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[2].strip():
			res['draw'] = int(response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[2].strip().split(' ')[0])
			res['nc'] = int(response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[2].strip().split(' ')[1].replace('(', ''))
		else:
			res['draw'] = int(response.css('span.b-content__title-record ::text').get().strip().split(':')[1].split('-')[2].strip())
			res['nc'] = 0


		for stats in response.css('ul li.b-list__box-list-item_type_block'):
			string_ = stats.xpath('string(.)').get().strip()
			string_ = string_.replace("\t", '')
			string_ = string_.replace("\n", '')
			string_ = string_.replace(" ", '')
			string_ = string_.replace("DA", 'D_A')
			string_ = string_.replace("DD", 'D_D')
			string_ = string_.replace(".A", '_A')
			string_ = string_.replace(".D", '_D')
			texts = string_.split(':')
			if len(texts) > 1:
				title = texts[0]
				content = texts[1]
				# print(title, content)
				if title == "Reach":
					if content == '--':
						res[title.lower().replace('.', '')] = 0
					else:
						res[title.lower().replace('.', '')] = int(content.strip().replace('.', '').replace('"', ''))
				if title in ["Height", "Weight", "STANCE", "DOB"]:
					res[title.lower().replace('.', '')] = content.strip().replace('.', '')
				if '%' in content:
					res[title.replace('.', '')] = float(content[:-1])/100
				try:
					if float(content.strip()) and '%' not in content:
						res[title.replace('.', '')] = float(content.strip())
				except Exception:
					pass
				if title.replace('.', '') == 'Sub_Avg' and float(content.strip()) == 0.0:
					res['Sub_Avg'] = 0.0
				results.append(res)		
			else:
				pass

		yield results
		
		# links = list(string.ascii_lowercase)

		# for link in links[1::]:
		# 	page = "http://ufcstats.com/statistics/fighters?char={}&page=all".format(link)
		# 	yield response.follow(page, callback=self.parse)

		open('Data/ufc_fighters.json', 'w+').write(json.dumps(results))
    
# with open('Data/ufc_fighters.json', 'w+') as f:
#     f.write(json.dumps(fighters))
#     f.close()

if __name__ == '__main__':
    # run scraper
    process = CrawlerProcess()
    process.crawl(UfcscraperSpider)
    process.start()