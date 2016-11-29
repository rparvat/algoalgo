##### THIS FILE CAN BE RUN TO GENERATE NEW INTERNAL REPS #####

from random import choice, random, randint

maps = ["MULT_BY", "ADD"]
filters = ["DIVISIBLE_BY", "LESS_THAN"]
reduces = ["SUM", "PROD", "AVG", "MIN"]

MAP_PROBABILITY = 0.8
FILTER_PROBABILITY = 0.8
REDUCE_PROBABILITY = 0.5

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

	rep = []

	def gen_num():
		return randint(2, 20)

	if hasMap:
		rep.append(choice(maps) + " " + str(gen_num()))
	if hasFilter:
		rep.append(choice(filters) + " " + str(gen_num()))
	if hasReduce:
		rep.append(choice(reduces))

	return ", ".join(rep)

with open("reps.txt", "w") as file:
	for i in range(1000):
		file.write(gen_rep())
		file.write("\n")




