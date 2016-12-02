##### THIS FILE CAN BE RUN TO GENERATE NEW INTERNAL REPS #####

from random import choice, random, randint

# These are the possible operators
maps_with_consts = ["MULT", "ADD"] # these require a constant
maps_without_consts = ["SQUARE", "INVERSE"] # these are self sufficient
filters = ["DIVIS_BY", "LESS_THAN"]
reduces = ["SUM", "PROD", "AVG", "MIN"]

# Probabilities each operator is in our rep. Independent for each.
MAP_PROBABILITY = 0.8
FILTER_PROBABILITY = 0.8
REDUCE_PROBABILITY = 0.5

# For convenience
MAP = "MAP"
FILTER = "FILTER"
REDUCE = "REDUCE"

# CONSTANTS FOR CONVERSION TO ENGLISH
# First: possible beginning stop phrases
beginning_choices = ["find", "evaluate", "take", "compute", "tell me"]
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

elements_choices = ["elements", "values", "numbers", "things"]

# FILTER: conversion
def filterToEnglish(filt):
	# get filt_type and num
	[filt_type, num] = filt.split()
	num = int(num)

	if filt_type == "DIVIS_BY":
		if num == 2:
			return "the even " + choice(elements_choices)
		else:
			return " ".join(["the", choice(elements_choices), "divisible by", str(num)])
	elif filt_type == "LESS_THAN":
		return " ".join(["the", choice(elements_choices), "less than", str(num)])
	else:
		raise NotImplementedError

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

	def gen_num():
		return randint(2, 5)

	if hasFilter:
		rep[FILTER] = choice(filters) + " " + str(gen_num())

	if hasMap:
		hasConstant = random() < MAP_PROBABILITY / 2.0
		if hasConstant:
			rep[MAP] = choice(maps_with_consts) + " " + str(gen_num())
		else:
			rep[MAP] = choice(maps_without_consts)

	if hasReduce:
		rep[REDUCE] = choice(reduces)

	return rep

def rep_to_string(rep):
	list_rep = [(each, rep[each]) for each in rep]
	list_rep.sort()
	list_rep = [each[1] for each in list_rep]
	return ", ".join(list_rep)


NUM_SAMPLES = 1000
with open("reps.txt", "w") as file:
	for i in range(NUM_SAMPLES):
		rep = gen_rep()
		file.write(rep_to_string(rep) + "\n")
		file.write(rep_to_english(rep) + "\n")
		file.write("\n")




