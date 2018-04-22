#!/usr/bin/env python3
# Tool that dumps all the information we need from the WikiExtraction parser [1] to create our own dataset into a file, such that it is later easily accessible
# [1]: https://github.com/attardi/wikiextractor

# Important: not efficient code, but it only has to run once on simple wiki, which is small

#  This tool relies on the output of the bash script 'extractDataWithWikiExtractor'

# The dumped dictionary will have the following form
# It will map from an article id to an article dictionary
# Each article dictionary will have the keys: meta, plain, links, lists

import argparse
import os
import pickle
from lxml import etree

# the dumped dictionary
articles = {}
		
xmlParser = etree.XMLParser(recover=True)

def removeXMLMarkups(article):
	article = article.strip();
	splits = article.split('\n', 1)
	assert(len(splits) == 2)
	assert(splits[0].startswith('<doc'))
	article = splits[1]
	
	splits = article.rsplit('\n', 1)
	assert(len(splits) == 2)
	assert(splits[1].startswith('</doc>'))
	article = splits[0]
	
	return article

def addArticleKeys(key, files):
	for fn in files:
		f = open(fn)
	
		currArticle = ''
		for line in f:
			currArticle += line
		
			if line.strip() == '</doc>':
				xmlArticle = etree.fromstring(currArticle, parser=xmlParser)
				assert(xmlArticle.attrib.keys() == ['id','url','title'])
			
				currid = int(xmlArticle.attrib['id'])
				assert(currid in articles)
			
				articles[currid][key] = removeXMLMarkups(currArticle)
				
				currArticle = ''

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, help="Directory, where the script 'extractDataWithWikiExtractor.sh' was run." + \
		"\nThus, folder that contains the directories: simplewiki-plain, simplewiki-lists, simplewiki-links", default='.')
	parser.add_argument('--output', type=str, help="Name and directory of outputfile.", default='wikiextraction_dump.pickle')
	args = parser.parse_args()
	
	inputDir = args.input
	output = args.output
	
	plainDir = os.path.join(inputDir, 'simplewiki-plain')
	linksDir = os.path.join(inputDir, 'simplewiki-links')
	listsDir = os.path.join(inputDir, 'simplewiki-lists')
	
	if not os.path.isdir(plainDir) or \
		not os.path.isdir(linksDir) or \
		not os.path.isdir(listsDir):
		raise(Exception('At least one of the following folders is not present in directory \'' + inputDir + \
			'\': simplewiki-plain, simplewiki-lists, simplewiki-links'))
			        
	getAllFilesInFolder = lambda folder : [os.path.join(root, name) \
		for root, dirs, files in os.walk(folder) \
		for name in files \
		if name.startswith(("wiki_"))]

	plainFiles = getAllFilesInFolder(plainDir)
	linksFiles = getAllFilesInFolder(linksDir)
	listsFiles = getAllFilesInFolder(listsDir)
	
	for fn in plainFiles:
		f = open(fn)
		
		currArticle = ''
		for line in f:
			currArticle += line
			
			if line.strip() == '</doc>':
				xmlArticle = etree.fromstring(currArticle, parser=xmlParser)	
				assert(xmlArticle.attrib.keys() == ['id','url','title'])			
				
				currid = int(xmlArticle.attrib['id'])
				assert(currid not in articles)
				
				articles[currid] = {}
				
				articles[currid]['meta'] = dict(xmlArticle.attrib)
				articles[currid]['plain'] = removeXMLMarkups(currArticle)
				
				currArticle = ''

	addArticleKeys('links', linksFiles)
	addArticleKeys('lists', listsFiles)

	pickle.dump(articles, open(output, "wb"))
	
	print('articles successfully dumped into ' + output)

