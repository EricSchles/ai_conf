import requests
import lxml.html
import time
from app.models import Ads
from app import db

MINUTE = 60
HOUR = 60

def scrape_search_results(url, base_url):
    print("scraped list of ads for url:",url)
    html = lxml.html.fromstring(requests.get(url).text)
    relative_links = html.xpath("//a[@class='result-title hdrlnk']/@href")
    return [base_url+elem for elem in relative_links]

def scrape_ad(url):
    print("scraping specific ad at url:",url)
    html = lxml.html.fromstring(requests.get(url).text)
    data = {}
    data["text_body"] = html.xpath("//section[@id='postingbody']")[0].text_content()
    post_id = [elem for elem in html.xpath("//p[@class='postinginfo']")
               if "post id" in elem.text_content()][0].text_content()
    data["post_id"] = post_id.replace("post id: ","")
    print("Saving Ad with post id:",data["post_id"])
    return data

def increment_url(base_url,offset):
    return base_url+"?s="+str(offset)

def save_to_db(data):
    if Ads.query.filter_by(post_id=data["post_id"]).all() == []:
        ad = Ads(data["text_body"],data["post_id"],"not prostitution")
        db.session.add(ad)
        db.session.commit()
    else:
        pass
    
def run():
    start_url = "https://newyork.craigslist.org/search/w4m"
    base_url = "https://newyork.craigslist.org"
    print("scraping craigslist ...")
    html = lxml.html.fromstring(requests.get(start_url).text)
    result = ' '.join(html.xpath("//span[@class='range']")[0].text_content().split())
    links_per_page = int(result.split(" ")[-1])
    total_links = int(html.xpath("//span[@class='totalcount']")[0].text_content())
    ad_data = []
    link_range = links_per_page
    url = start_url
    while True:
        try:
            ads = scrape_search_results(url, base_url)
            for ad in ads:
                save_to_db( scrape_ad(ad) )
            url = increment_url(url, link_range)
            link_range += links_per_page
            if link_range > total_links:
                print("sleeping...")
                time.sleep(MINUTE * HOUR * 5)
                url = start_url
        except:
            print("got weird error...")
            print("sleeping for a while to wait for new content to become available")
            time.sleep(MINUTE * HOUR)
