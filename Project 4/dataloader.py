import csv
import string

def fetch_untokenized(filename):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		data = []
		for index, row in enumerate(reader):
			if index == 0:
				continue
			ending1 = row[5]
			ending2 = row[6]
			gold = int(row[7])
			data.append((row[1], row[2], row[3], row[4], ending1, ending2, gold))
		return data

def fetch_test_untokenized(filename):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		data = []
		for index, row in enumerate(reader):
			if index == 0:
				continue
			sid = row[0]
			ending1 = row[5]
			ending2 = row[6]
			data.append((sid, row[1], row[2], row[3], row[4], ending1, ending2))
		return data