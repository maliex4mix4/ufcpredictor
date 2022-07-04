import scrapy, string, json, os
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from scrapy.http import HtmlResponse, Request
from scrapy_selenium import SeleniumRequest

class UfcscraperSpider(scrapy.Spider):
	name = 'ufcscraper'

	start_urls = [
		'http://ufcstats.com/statistics/fighters?char=a&page=all'
	]

	def parse(self, response):
		results = []
		for user_info in response.css(".b-statistics__table-row")[2::]:
		
			result = user_info.css("td:nth-child(1) a::attr(href)").get().strip()
			
			results.append(result)

		yield result
		
		links = list(string.ascii_lowercase)

		for link in links[1::]:
			page = "http://ufcstats.com/statistics/fighters?char={}&page=all".format(link)
			yield response.follow(page, callback=self.parse)

		for url in results:
			yield Request(url=url, callback=self.parse_individual, dont_filter=True)

	def parse_individual(self, response, **kwargs):
		res = {}
		# keys = response.css('ul li.b-list__box-list-item_type_block i ::text').getall()
		# values = response.css('ul li.b-list__box-list-item_type_block ::text').getall()

		# def cleaned_key_vals(key, vals):
		# 	k = [k.strip() for k in key]
		# 	v = [v.strip() for v in vals]
		# 	vl = [v_ for v_ in v if v_ not in k ]

		# 	return k, vl

		# keys, values = cleaned_key_vals(keys, values)

		# for x in keys:
		# 	if len(x) < 1:
		# 		pass
		# 	else:
		# 		try:
		# 			res[x] = values[keys.index(x)]
		# 		except:
		# 			res[x] = None

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
				if title == "Reach" and content == '--':
					res[title.lower().replace('.', '')] = 0
				else:
					res[title.lower().replace('.', '')] = int(content.strip().replace('.', '').replace('"', ''))
				if title in ["Height", "Weight", "STANCE", "DOB"]:
					res[title.lower().replace('.', '')] = content.strip().replace('.', '').replace('\'', '')
				if '%' in content:
					res[title.replace('.', '')] = float(content[:-1])/100
				try:
					if float(content.strip()) and '%' not in content:
						res[title.replace('.', '')] = float(content.strip())
				except Exception:
					pass
				if title.replace('.', '') == 'Sub_Avg':
					print("Subimtion average", float(content.strip()))
						
			else:
				pass
		print(res)

if __name__ == '__main__':
    # run scraper
    process = CrawlerProcess()
    process.crawl(UfcscraperSpider)
    process.start()