#!/usr/bin/env python
##### THIS FILE CAN BE RUN TO GENERATE NEW INTERNAL REPS #####

from random import choice, random, randint

# These are the possible operators
maps_with_consts = ["MULT", "ADD"] # these require a constant
maps_without_consts = ["SQUARE", "INVERSE"] # these are self sufficient
filters = ["DIVIS_BY", "LESS_THAN", "GREATER_THAN", "EQUAL"]
reduces = ["SUM", "PROD", "AVG", "MIN"]

# Probabilities each operator is in our rep. Independent for each.
MAP_PROBABILITY = 0.8
FILTER_PROBABILITY = 0.8
REDUCE_PROBABILITY = 0.5
NOT_PROBABILITY = 0.5

# For convenience
MAP = "MAP"
FILTER = "FILTER"
REDUCE = "REDUCE"

# CONSTANTS FOR CONVERSION TO ENGLISH
# First: possible beginning stop phrases
beginning_choices = ["find", "evaluate", "take", "compute"]
BEGINNING_PROBABILITY = 0.8

# REDUCE: conversion to English. Standalone means no beginning choice.
REDUCE_STANDALONE_PROBABILITY = 0.25
reduce_dict = {
	"SUM": ["the sum"],
	"PROD": ["the product"],
	"AVG": ["the mean", "the average", "the mean value"],
	"MIN": ["the min", "the min value", "the minimum", "the minimum value"]
}
reduce_dict_standalone = {
	"SUM": ["sum", "add up", "add all"],
	"PROD": ["multiply", "multiply all"],
	"AVG": ["average"],
	"MIN": ["find the min of"]
}
def reduceToEnglish(red):
	if random() < REDUCE_STANDALONE_PROBABILITY:
		return choice(reduce_dict_standalone[red])
	if random() < BEGINNING_PROBABILITY:
		return " ".join([choice(beginning_choices), choice(reduce_dict[red]), "of"])
	return " ".join([choice(reduce_dict[red]), "of"])

elements_choices = ["elements", "values", "numbers", "things", "array", "list"]

# FILTER: conversion
def filterToEnglish(filt):
	# get filt_type and num
	[filt_type, num] = filt.split()
	first_words = ["the", choice(elements_choices)]
	if random() < 0.5:
		first_words.append("that are")
	second_words = []

	hasNot = random() < NOT_PROBABILITY
	if hasNot:
		second_words.append("not")

	if filt_type == "DIVIS_BY":
		second_words.append("divisible by")
	elif filt_type == "LESS_THAN":
		second_words.append(choice(["less than", "smaller than"]))
	elif filt_type == "EQUAL":
		second_words.append("equal to")
	elif filt_type == "GREATER_THAN":
		second_words.append(choice(["greater than", "bigger than"]))
	else:
		raise NotImplementedError

	return " ".join(first_words + second_words + [num])

#MAP: conversion
m_wo_consts_dict = {
	"SQUARE": ["the squares of"],
	"INVERSE": ["the inverses of", "the reciprocals of"]
}
def mapToEnglish(m):
	if m in maps_without_consts:
		return choice(m_wo_consts_dict[m])
	# otherwise, it's a map with a constant
	[m, num] = m.split()
	if m == "MULT":
		return " ".join([num, "times"])
	elif m == "ADD":
		return " ".join([num, "plus"])
	else:
		raise NotImplementedError


# Now: will turn any general rep into english.
def rep_to_english(rep):
	english = ""

	if REDUCE in rep:
		english += reduceToEnglish(rep[REDUCE])
	else:
		english += choice(beginning_choices)

	if MAP in rep:
		english += " " + mapToEnglish(rep[MAP])

	if FILTER in rep:
		english += " " + filterToEnglish(rep[FILTER])
	else:
		english += " the " + choice(elements_choices)

	return english

def genNumber():
	return str(choice(range(1, 6)))

# There are a few kinds of internal reps without considering reducers:
#	1. NONE
#	2. FILTER
#	3. MAP
#	4. FILTER, MAP
#	we choose not to consider a map before a filter, because this is unlikely to be needed
#	IMPORTANT: any of the above patterns can be followed by a reduce

def gen_rep():
	hasMap = random() < MAP_PROBABILITY
	hasFilter = random() < FILTER_PROBABILITY
	hasReduce = random() < REDUCE_PROBABILITY

	rep = {}

	if hasFilter:
		rep[FILTER] = " ".join([choice(filters), genNumber()])

	if hasMap:
		hasConstant = random() < 1.0 / 2.0
		if hasConstant:
			rep[MAP] = " ".join([choice(maps_with_consts), genNumber()])
		else:
			rep[MAP] = choice(maps_without_consts)

	if hasReduce:
		rep[REDUCE] = choice(reduces)

	return rep

def rep_to_string(rep):
	list_rep = [(each, rep[each]) for each in rep]
	list_rep.sort()
	list_rep = [each[1] for each in list_rep]
	return " ".join(list_rep)


#### NOW: GENERATION OF COMPLEX REPS

def complex_rep_to_string(complex_rep):
	list_rep = []
	for each_list in complex_rep:
		for operator in complex_rep:
			list_rep.append(operator)
	return " ".join(list_rep)

complex_maps_without_consts = ["SORT"] + maps_without_consts
complex_maps_with_conts = [] + maps_with_consts
complex_maps = complex_maps_with_conts + complex_maps_without_consts

def gen_array_ref():
	return choice(["it", "them", "the " + choice(elements_choices)])

def gen_complex_filter():
	return " ".join([choice(filters), genNumber()])

def complex_filter_to_english(filter_op):
	return " "

def gen_complex_map():
	our_map = choice(complex_maps)
	if our_map in complex_maps_without_consts:
		return our_map
	return " ".join([our_map, genNumber()])

def complex_map_to_english(m):
	if m == "SORT":
		if random() < 0.5:
			return "sort " + gen_array_ref()
		return " ".join([choice(beginning_choices), "the sorted", choice(elements_choices)])
	elif m == "SQUARE" or m == "INVERSE":
		verb = "square" if m == "SQUARE" else choice(["invert", "reciprocate"])
		if random() < 0.5:
			return " ".join([verb, gen_array_ref()])
		return " ".join([choice(beginning_choices), choice(m_wo_consts_dict[m]), gen_array_ref()])
	else:
		[m, num] = m.split()
		if random() < 0.5:
			variant = "plus" if m == "ADD" else "times"
			map_description = [gen_array_ref(), variant, num]
			map_description = map_description[::-1] if random() < 0.5 else map_description
			map_description = [choice(beginning_choices)] + map_description
			return " ".join(map_description)
		else:
			map_description = None
			if m == "ADD":
				map_description = ["add", num, "to", gen_array_ref()]
			else:
				map_description = ["multiply", gen_array_ref(), "by", num]
			return " ".join(map_description)

def gen_if():
	# format: if (filter) then (map1, ..., mapn) else (map1, ..., mapn) endif
	if_list = ["IF"]
	if_filter = gen_complex_filter()
	if_reducer = choice(reduces)
	if_list = if_list + [if_filter, if_reducer]

	then_list = ["THEN"]
	then_map = gen_complex_map()
	then_list.append(then_map)

	else_list = ["ELSE"]
	else_map = gen_complex_map()
	else_list.append(else_map)

	endif_list = ["ENDIF"]

	return if_list + then_list + else_list + endif_list

def if_to_english(ops):
	then_index = ops.index("THEN")
	else_index = ops.index("ELSE")
	endif_index = -1

	if_list = ops[:then_index]
	then_list = ops[then_index: else_index]
	else_list = ops[else_index: endif_index]

	def else_to_english(else_list):
		beginning = choice(["otherwise", "if not", "else"])



# def gen_complex_rep():
# 	# so for each of the first few sentences, want only maps and filters.
# 	# each sentences: represented as sub-lists.



if __name__ == "__main__":
	NUM_SAMPLES = 10000
	with open("reps.txt", "w") as file:
		for i in range(NUM_SAMPLES):
			rep = gen_rep()
			file.write(rep_to_string(rep) + ",")
			file.write(rep_to_english(rep) + "\n")
