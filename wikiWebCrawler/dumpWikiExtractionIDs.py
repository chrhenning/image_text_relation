#!/usr/bin/env python3
# Tool that dumps all the (ids, titles, urls) of SimpleWiki articles into a pickle file. It reads this information from the output of the WikiExtraction parser [1]
# [1]: https://github.com/attardi/wikiextractor

import argparse
import os
import random
import pickle
from lxml import etree

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('inputDir', type=str, help="Directory, where the wikiExtraction output was written to")
	parser.add_argument('--output', type=str, help="Name and directory of outputfile.", default='wikiextraction_id_dump.pickle')
	args = parser.parse_args()
	
	inputDir = args.inputDir
	output = args.output
        
	files = [os.path.join(root, name)
		for root, dirs, files in os.walk(inputDir)
		for name in files
		if name.startswith(("wiki_"))]
		
	articleList = []
		
	parser = etree.XMLParser(recover=True)
	
	for fn in files:
		f = open(fn)
		
		currArticle = ''
		for line in f:
			currArticle += line
			
			if line.strip() == '</doc>':
				xmlArticle = etree.fromstring(currArticle, parser=parser)
				currArticle = ''

				articleList.append(dict(xmlArticle.attrib))
				
	random.shuffle(articleList)

	pickle.dump(articleList, open(output, "wb"))
	
	print('article list successfully dumped into ' + output)

