from django.test import TestCase
import urllib2, os

image_path = '/home/chaopengz/larrypage.jpg'
url = "http://flower.chaopengz.com/upload/"
length = os.path.getsize(image_path)
png_data = open(image_path, "rb")
request = urllib2.Request(url, data=png_data)
request.add_header('Cache-Control', 'no-cache')
request.add_header('Content-Length', '%d' % length)
request.add_header('Content-Type', 'image/png')
res = urllib2.urlopen(request).read().strip()
print res
