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
		keys = response.css('ul li.b-list__box-list-item_type_block i ::text').getall()
		values = response.css('ul li.b-list__box-list-item_type_block ::text').getall()

		def cleaned_key_vals(key, vals):
			k = [k.strip() for k in key]
			v = [v.strip() for v in vals]
			vl = [v_ for v_ in v if v_ not in k ]

			return k, vl

		keys, values = cleaned_key_vals(keys, values)

		for x in keys:
			if len(x) < 1:
				pass
			else:
				try:
					res[x] = values[keys.index(x)]
				except:
					res[x] = None
		print(res, values)

if __name__ == '__main__':
    # run scraper
    process = CrawlerProcess()
    process.crawl(UfcscraperSpider)
    process.start()