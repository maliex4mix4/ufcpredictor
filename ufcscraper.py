import scrapy, string, json, os
from scrapy.crawler import CrawlerProcess

class UfcscraperSpider(scrapy.Spider):
	name = 'ufcscraper'

	start_urls = [
		'http://ufcstats.com/statistics/fighters?char=a&page=all'
	]

	def parse(self, response):
		results = []
		for user_info in response.css(".b-statistics__table-row")[2::]:			
			result = {
				"fname": user_info.css("td:nth-child(1) a::text").get(),
				"lname": user_info.css("td:nth-child(2) a::text").get(),
				"nname": user_info.css("td:nth-child(3) a::text").get(),
				"height": user_info.css("td:nth-child(4)::text").get().strip(),
				"weight": user_info.css("td:nth-child(5)::text").get().strip(),
				"reach": user_info.css("td:nth-child(6)::text").get().strip(),
				"stance": user_info.css("td:nth-child(7)::text").get().strip(),
				"win": user_info.css("td:nth-child(8)::text").get().strip(),
				"lose": user_info.css("td:nth-child(9)::text").get().strip(),
				"draw": user_info.css("td:nth-child(10)::text").get().strip()
			}
			results.append(result)

		yield result

		links = list(string.ascii_lowercase)

		for link in links[1::]:
			page = "http://ufcstats.com/statistics/fighters?char={}&page=all".format(link)
			yield response.follow(page, callback=self.parse)

		page = response.url.split("&")[0][-1]
		filename = f'fighters-{page}.json'

		path = os.getcwd()
		new_path = path.split("\\")[0:-2]
		# new_path = ("\\").join(new_path) + "\\fighter"
		new_path = path + "\\Data\\fighter"
		if not os.path.exists(new_path):
			os.makedirs(new_path)
		completeName = os.path.join(new_path, filename)

		with open(completeName, 'w+') as f:
			f.write(json.dumps(results, indent=2))
			f.close()
		self.log(f'Saved file {completeName}')


		# for key, value in enumerate(result):
			# 	for k, v in value.items():
			# 		if v == "":
			# 			result[idx][k] = 'DEFAULT'

# main driver
if __name__ == '__main__':
    # run scraper
    process = CrawlerProcess()
    process.crawl(UfcscraperSpider)
    process.start()