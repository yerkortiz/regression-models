package utility

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"time"
)

type twitData struct {
	CreatedAt string `json:"created_at"`
	ID        int    `json:"id"`
	Unix      int64
}
type parsedData struct {
	Unix         int64
	Country      string
	DisasterType string
	Quantity     int
}

func evalErr(err error) {
	if err != nil {
		log.Println(err)
	}
}
func readRaw(fileName string) []twitData {
	file, err := os.Open(fileName)
	evalErr(err)
	defer file.Close()
	var result []twitData
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		var obj twitData
		err := json.Unmarshal(json.RawMessage(line), &obj)
		evalErr(err)
		date, err := time.Parse(time.RubyDate, obj.CreatedAt)
		evalErr(err)
		obj.Unix = date.Unix()
		result = append(result, obj)
	}
	err = scanner.Err()
	evalErr(err)
	file.Close()
	return result
}
func parseRaw(result []twitData, country string, disasterType string) []parsedData {
	sort.Slice(result, func(i, j int) bool {
		return result[i].Unix < result[j].Unix
	})
	var resultGrouped []parsedData
	c := 0
	current := (result[0].Unix - result[0].Unix%60)
	max := (result[len(result)-1].Unix + 60 - result[len(result)-1].Unix%60)
	for i := 0; i < len(result) && current < (max+60); current += 60 {
		for ; i < len(result) && current > result[i].Unix; i++ {
			c++
			fmt.Println(result[i].ID, " va al saco")
		}
		temp := parsedData{current, country, disasterType, c}
		resultGrouped = append(resultGrouped, temp)
		c = 0
	}
	return resultGrouped
}
func writeCsv(parsedData []parsedData, outFilePath string) {
	var parsedResult [][]string
	for _, value := range parsedData {
		var temp []string
		temp = append(temp, value.Country)
		temp = append(temp, strconv.FormatInt(value.Unix, 10))
		temp = append(temp, value.DisasterType)
		temp = append(temp, strconv.Itoa(value.Quantity))
		parsedResult = append(parsedResult, temp)
	}
	fileCsv, err := os.Create(outFilePath)
	evalErr(err)
	defer fileCsv.Close()
	writer := csv.NewWriter(fileCsv)
	defer writer.Flush()
	for _, value := range parsedResult {
		err := writer.Write(value)
		evalErr(err)
	}
}
func main() {
	inFilePath := "/Users/yerko/codes/twitter-traffic-predict/raw_data/nepal_2015.jsonl"
	country := "Nepal"
	disasterType := "Earthquake"
	outFilePath := "/Users/yerko/codes/twitter-traffic-predict/datasets/dataset.csv"
	result := readRaw(inFilePath)
	resultGrouped := parseRaw(result, country, disasterType)
	writeCsv(resultGrouped, outFilePath)
}
