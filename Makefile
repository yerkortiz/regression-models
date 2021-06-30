parse:
	go run ./utility/parse_raw.go
train:
	python3 rnn.py 
regression:
	python3 regression.py