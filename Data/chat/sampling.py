import json

file_name = 'chatbot.txt'
k = 2
f = open(file_name)
i = 0
for line in f:
	i += 1
	if not line.isspace() and i % 2 == 0:
		print line.strip().lower()
f.close()