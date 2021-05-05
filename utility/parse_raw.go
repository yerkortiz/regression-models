package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
)

func main() {
	/*
		fileName := os.Args[1]
		country := os.Args[2]
		disasterType := os.Args[3]
		outPath := os.Args[4]
	*/
	fileName := "/Users/yerko/codes/twitter-traffic-predict/raw_data/example.jsonl"
	country := "Chile"
	disasterType := "Earthquake"
	outPath := "/Users/yerko/codes/twitter-traffic-predict/datasets"
	outFileName := "dataset.csv"
	fmt.Println("Parsing:", fileName)
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	type twitData struct {
		CreatedAt    string `json:"created_at"`
		ID           int    `json:"id"`
		Unix         int64
		Country      string
		DisasterType string
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
		obj.Country = country
		obj.DisasterType = disasterType
		result = append(result, obj)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	var parsedResult [][]string
	//headerCsv := []string{"Country", "Unix", "Disaster", "ID"}
	//parsedResult = append(parsedResult, headerCsv)
	for _, value := range result {
		var temp []string
		temp = append(temp, value.Country)
		temp = append(temp, strconv.FormatInt(value.Unix, 10))
		temp = append(temp, value.DisasterType)
		temp = append(temp, strconv.Itoa(value.ID))
		parsedResult = append(parsedResult, temp)
	}
	fileCsv, err := os.Create(outPath + "/" + outFileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	writer := csv.NewWriter(fileCsv)
	defer writer.Flush()
	for _, value := range parsedResult {
		err := writer.Write(value)
		if err != nil {
			log.Fatal(err)
		}
	}
}
