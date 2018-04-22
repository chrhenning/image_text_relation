#!/usr/bin/env python3
# Outputs annotation statistics for the wiki dataset

import argparse
import os
import json

# color codes (ANSI escape codes)
RED='\033[0;31m'
GREEN='\033[0;32m'
LIGHT_GREEN='\033[1;32m'
ORANGE='\033[1;33m'
NC='\033[0m' # No Color

def printMI(mi, prependingSpaces=0):
	strPrec = 2

	heading = ['Label', '0', '1', '2', '3', '4', '5', '6', '7']
	row_format = ' ' * prependingSpaces + "# {:<15} | " + "{:>15}" * (len(heading) - 1)
	
	total = 0
	for v in mi.values():
		total += v
	
	totals = []
	perc = []
	for i in range(8):
		totals.append(mi[i])
		perc.append(round(100*(float(mi[i])/total),strPrec))
	
	print(row_format.format(*heading))
	#print(row_format.format('Total', round(mi[0],strPrec), round(mi[1],strPrec), round(mi[2],strPrec), round(mi[3],strPrec), round(mi[4],strPrec), round(mi[5],strPrec), round(mi[6],strPrec), round(mi[7],strPrec)))
	print(' ' * prependingSpaces + "# " + "-"*15  +"-|-" + "-" * 15 * (len(heading) - 1))
	print(row_format.format('Total', *totals))
	print(row_format.format('Percentage', *perc))
	
def printSC(sc, prependingSpaces=0):
	strPrec = 2

	heading = ['Label', '-1.0', '-0.5', '0.0', '0.5', '1.0']
	row_format = ' ' * prependingSpaces + "# {:<15} | " + "{:>15}" * (len(heading) - 1)
	
	total = 0
	for v in sc.values():
		total += v
	
	totals = []
	perc = []
	for i in range(5):
		l = float(i-2)/2
		totals.append(sc[l])
		perc.append(round(100*(float(sc[l])/total),strPrec))
	
	print(row_format.format(*heading))
	print(' ' * prependingSpaces + "# " + "-"*15  +"-|-" + "-" * 15 * (len(heading) - 1))
	#print(row_format.format('Total', round(mi[0],strPrec), round(mi[1],strPrec), round(mi[2],strPrec), round(mi[3],strPrec), round(mi[4],strPrec), round(mi[5],strPrec), round(mi[6],strPrec), round(mi[7],strPrec)))
	print(row_format.format('Total', *totals))
	print(row_format.format('Percentage', *perc))
	
	mean = 0
	meani = total / 2
	meann = 0
	for i in range(5):
		l = float(i-2)/2		
		meann += sc[l]
		if meann >= meani:
			mean = l
			break
	
	average = 0.0
	for k,v in sc.items():
		average += k*v
		
	average /= total	
	
	print('### Mean Semantic Correlation: ' + str(mean))
	print('### Avg Semantic Correlation: ' + str(round(average,strPrec)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help="The file contaning wiki annotations (.jsonl file).")
	#parser.add_argument('--output', type=str, help="Name of the jsonl file (will be th same filename with extension .jsonl if not specified).")
	args = parser.parse_args()
	
	inFile = args.input
	#output = args.output
	

	
	if not os.path.isfile(inFile):
		raise(Exception('The file' + inFile + 'does not exists'))
	
	'''
	if output == None:
		filename, _ = os.path.splitext(inFile)
		output = filename + '.jsonl'
		print('Output file will be: ' + output)
		if os.path.isfile(output):
			r = input('Output file already exists. Would you like to overwrite it? (y/n): ')
			if r.lower() != 'y':					
				raise(Exception('No output file specified'))
	'''
	
	fi = open(inFile, 'r')
	
	annos = []
	
	for line in fi:
		annos.append(json.loads(line))
	
	fi.close()
	
	print('### Total number of annotated image-text samples: ' + str(len(annos)))
	
	# group annotations by image typ
	categories = {}
	
	# total counts of all MI values
	mi = {}
	for i in range(8):
		mi[i] = 0
	# total counts of all SC values
	sc = {}
	for i in range(5):
		sc[float(i-2)/2] = 0
	# total number of invalid samples
	invalid = 0
	# total number of annos with snippets
	snippets = 0
	
	for a in annos:
		if a['valid'] and a['mi'] >= 0 and a['sc'] >= -1:
			if a['type'] not in categories:
				categories[a['type']] = []
			categories[a['type']].append(a)

			mi[a['mi']] += 1
			sc[a['sc']] += 1
			
			if len(a['snippets']) > 0:
				snippets += 1
			
		else:
			invalid += 1
			
	print('### Distribution of MI labels:')
	printMI(mi,0)
	print('### Distribution of SC labels:')
	printSC(sc,0)
	print('### Number of invalid samples: ' + str(invalid))
	print('### Number of samples marked with highly correlated snippets: ' + str(snippets))
	print()
	print(ORANGE + '### Statistics for each image type' + NC)
	
	for cat, annos in categories.items():
		print()
		print('### Statistics for image type: ' + LIGHT_GREEN + cat + NC)
		print('### Total number of annotated image-text samples: ' + str(len(annos)))
		# total counts of all MI values
		mi = {}
		for i in range(8):
			mi[i] = 0
		# total counts of all SC values
		sc = {}
		for i in range(5):
			sc[float(i-2)/2] = 0
		# total number of annos with snippets
		snippets = 0
	
		for a in annos:		
			mi[a['mi']] += 1
			sc[a['sc']] += 1
		
			if len(a['snippets']) > 0:
				snippets += 1

			
		print('### Distribution of MI labels:')
		printMI(mi,0)
		print('### Distribution of SC labels:')
		printSC(sc,0)
		print('### Number of samples marked with highly correlated snippets: ' + str(snippets))
	print()
	
	#fo = open(output, 'w')
	#fo.close()

