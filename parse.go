package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	fileName := os.Args[1]
	fmt.Println("Parsing:", fileName)
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	type twitData struct {
		CreatedAt string `json:"created_at"`
		ID        int    `json:"id"`
		Unix      int64  `json:"unix"`
	}
	var result []twitData
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		var obj twitData
		err := json.Unmarshal(json.RawMessage(line), &obj)
		if err != nil {
			log.Println(err)
		}
		date, err := time.Parse(time.RubyDate, obj.CreatedAt)
		if err != nil {
			log.Println(err)
		}
		obj.Unix = date.Unix()
		result = append(result, obj)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	for _, x := range result {
		fmt.Println(x)
	}
}
