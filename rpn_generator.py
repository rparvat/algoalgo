#!/usr/bin/env python
from random import choice, random

MIN_SAMPLE_LEN = 2
MAX_SAMPLE_LEN = 10
NUM_SAMPLES = 100000

def get_value():
	rand = random()
	if rand < 0.2:
		nrstatement = get_nrstatement()
		reducer = get_reducer()
		algo = nrstatement[0] + reducer[0]
		rand = random()
		if rand < 0.5:
			text = nrstatement[1] + choice([['and'], []]) + choice([['then'], []]) + reducer[1] + choice([[], ['of', 'the', choice(['elements', 'numbers', 'members'])]]) + choice([[], choice([['in'], ['of']]) + choice([['it'], ['the', 'array']])])
		else:
			text = reducer[1] + choice([[], ['of', 'the', choice(['elements', 'numbers', 'members'])]]) + choice([[], choice([['in'], ['of']]) + choice([[], ['the', 'array']])]) + nrstatement[1]
		return (algo, text)
	elif rand < 0.4:
		algo = ['2']
		text = choice([['2'], ['two']])
		return (algo, text)
	elif rand < 0.6:
		algo = ['3']
		text = choice([['3'], ['three']])
		return (algo, text)
	elif rand < 0.8:
		algo = ['4']
		text = choice([['4'], ['four']])
		return (algo, text)
	else:
		algo = ['5']
		text = choice([['5'], ['five']])
		return (algo, text)

def get_elemmap():
	rand = random()
	if rand < 0.2:
		elemmap1 = get_elemmap()
		elemmap2 = get_elemmap()
		condition = get_condition()
		algo = elemmap1[0] + elemmap2[0] + condition[0] + ['IF']
		rand = random()
		if rand < 0.5:
			text = ['if'] + condition[1] + choice([['is', 'true'], []]) + choice([['then'], []]) + elemmap1[1] + choice([['otherwise'], ['else'], ['else', 'then'], ['if', 'not'], ['if', 'not', 'then']]) + elemmap2[1] + choice([['instead'], []])
		elif rand < 0.75:
			text = elemmap1[1] + ['if'] + condition[1] + choice([['is', 'true'], []]) + choice([['otherwise'], ['else'], ['else', 'then'], ['if', 'not'], ['if', 'not', 'then']]) + elemmap2[1] + choice([['instead'], []])
		else:
			text = elemmap1[1] + ['if'] + condition[1] + choice([['is', 'true'], []]) + elemmap2[1] + choice([['instead'], []]) + choice([['otherwise'], ['else'], ['if', 'not']])
		return  (algo, text)
	elif rand < 0.4:
		value = get_value()
		algo = value[0] + ['ADD']
		text = choice([['add'] + value[1] + ['to'], value[1] + ['plus']]) + choice([[], ['the', 'elements', 'of']])
		return (algo, text)
	elif rand < 0.6:
		value = get_value()
		algo = value[0] + ['MULT']
		text = value[1] + ['times'] + choice([[], ['the', 'elements', 'of']])
		return (algo, text)
	elif rand < 0.8:
		algo = ['SQUARE']
		text = ['the', 'squares', 'of'] + choice([[], ['the', 'elements', 'of']])
		return (algo, text)
	else:
		algo = ['RECIP']
		text = ['the', 'reciprocals', 'of'] + choice([[], ['the', 'elements', 'of']])
		return (algo, text)

def get_nrstatement():
	rand = random()
	if rand < 0.3:
		nrstatement = get_nrstatement()
		conditional = get_conditional()
		algo = nrstatement[0] + conditional[0] + ['FILTER']
		text = ['the', 'elements', 'of'] + nrstatement[1] + choice([[], ['that', 'are']]) + conditional[1]
		return (algo, text)
	elif rand < 0.6:
		nrstatement = get_nrstatement()
		elemmap = get_elemmap()
		algo = nrstatement[0] + elemmap[0] + ['MAP']
		text = elemmap[1] + nrstatement[1]
		return (algo, text)
	else:
		algo = ['INPUT']
		text = ['the', 'input'] + choice([[], ['array']])
		return (algo, text)

def get_reducer():
	rand = random()
	if rand < 0.2:
		algo = ['SUM']
		rand = random()
		if rand < 0.5:
			text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + ['the'] + choice([['sum'], ['total']])
		else:
			text = choice([['sum'], ['add'], ['total']]) + choice([[], ['up']]) + choice([[], ['all']])
		return (algo, text)
	elif rand < 0.4:
		algo = ['PROD']
		rand = random()
		if rand < 0.5:
			text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + ['the'] + ['product']
		else:
			text = ['multiply'] + choice([[], ['all']])
		return (algo, text)
	elif rand < 0.6:
		algo = ['AVG']
		rand = random()
		if rand < 0.5:
			text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + ['the'] + choice([['average'], ['mean']])
		else:
			text = ['average'] + choice([[], ['up']]) + choice([[], ['all']])
		return (algo, text)
	elif rand < 0.8:
		algo = ['MIN']
		text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + ['the'] + choice([['min'], ['minimum'], ['smallest'], ['lowest'], ['least']]) + choice([[], ['element'], ['number'], ['member']])
		return (algo, text)
	else:
		algo = ['MAX']
		text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + ['the'] + choice([['max'], ['maximum'], ['largest'], ['highest'], ['greatest']]) + choice([[], ['element'], ['number'], ['member']])
		return (algo, text)

def get_conditional():
	rand = random()
	if rand < 0.1:
		conditional = get_conditional()
		algo = conditional[0] + ['NOT']
		text = ['not'] + conditional[1]
		return (algo, text)
	elif rand < 0.2:
		conditional1 = get_conditional()
		conditional2 = get_conditional()
		algo = conditional1[0] + conditional2[0] + ['OR']
		text = choice([[], ['either']]) + conditional1[1] + ['or'] + conditional2[1]
		return (algo, text)
	elif rand < 0.4:
		value = get_value()
		algo = value[0] + ['DIVIS_BY']
		text = choice([['divisible', 'by'], ['a', 'multiple', 'of']]) + value[1]
		return (algo, text)
	elif rand < 0.6:
		value = get_value()
		algo = value[0] + ['LESS_THAN']
		text = choice([['less', 'than'], ['smaller', 'than']]) + value[1]
		return (algo, text)
	elif rand < 0.8:
		value = get_value()
		algo = value[0] + ['GREATER_THAN']
		text = choice([['greater', 'than'], ['bigger', 'than'], ['larger', 'than']]) + value[1]
		return (algo, text)
	else:
		value = get_value()
		algo = value[0] + ['EQUAL']
		text = choice([['equals'], ['equal', 'to'], ['the', 'same', 'as']]) + value[1]
		return (algo, text)

def get_condition():
	value = get_value()
	conditional = get_conditional()
	algo = value[0] + conditional[0]
	text = value[1] + choice([[], ['is']]) + conditional[1]
	return (algo, text)

def get_statement():
	rand = random()
	if rand < 0.2:
		statement1 = get_statement()
		statement2 = get_statement()
		condition = get_condition()
		algo = statement1[0] + statement2[0] + condition[0] + ['IF']
		rand = random()
		if rand < 0.5:
			text = ['if'] + condition[1] + choice([['is', 'true'], []]) + choice([['then'], []]) + statement1[1] + choice([['otherwise'], ['else'], ['else', 'then'], ['if', 'not'], ['if', 'not', 'then']]) + statement2[1] + choice([['instead'], []])
		elif rand < 0.75:
			text = statement1[1] + ['if'] + condition[1] + choice([['is', 'true'], []]) + choice([['otherwise'], ['else'], ['else', 'then'], ['if', 'not'], ['if', 'not', 'then']]) + statement2[1] + choice([['instead'], []])
		else:
			text = statement1[1] + ['if'] + condition[1] + choice([['is', 'true'], []]) + statement2[1] + choice([['instead'], []]) + choice([['otherwise'], ['else'], ['if', 'not']])
		return  (algo, text)
	elif rand < 0.5:
		nrstatement = get_nrstatement()
		reducer = get_reducer()
		algo = nrstatement[0] + reducer[0]
		rand = random()
		if rand < 0.5:
			text = nrstatement[1] + choice([['and'], []]) + choice([['then'], []]) + reducer[1] + choice([[], ['of', 'the', choice(['elements', 'numbers', 'members'])]]) + choice([[], choice([['in'], ['of']]) + choice([['it'], ['the', 'array']])])
		else:
			text = reducer[1] + choice([[], ['of', 'the', choice(['elements', 'numbers', 'members'])]]) + choice([[], choice([['in'], ['of']]) + choice([[], ['the', 'array']])]) + nrstatement[1]
		return (algo, text)
	else:
		nrstatement = get_nrstatement()
		algo = nrstatement[0]
		text = choice([['take'], ['find'], ['evaluate'], ['figure', 'out'], ['calculate'], ['compute']]) + nrstatement[1]
		return (algo, text)

def get_sample():
	return get_statement()

def is_sample_valid(sample):
	return len(sample[0]) <= MAX_SAMPLE_LEN and len(sample[0]) >= MIN_SAMPLE_LEN

if __name__ == "__main__":
	samples = []
	print('Generating training data')
	while len(samples) < NUM_SAMPLES:
		sample = get_sample()
		if is_sample_valid(sample):
			samples.append(sample)

	print('Writing samples to file')
	with open('rpn.txt', 'w') as f:
		for sample in samples:
			f.write(' '.join(sample[0]) + ',' +  ' '.join(sample[1]) + '\n')
