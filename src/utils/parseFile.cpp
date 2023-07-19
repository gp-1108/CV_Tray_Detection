#include "parseFile.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

void parseFile(const std::string& filename, std::vector<std::vector<int>>& scalars) {
  std::ifstream inputFile(filename);
  
  if (!inputFile) {
    std::cout << "Failed to open the file." << std::endl; //TODO sistemare il return
  }
  
  std::string line;
  
  while (std::getline(inputFile, line)) {
    std::stringstream ss(line);
    std::string idString, valuesString;

    // Read the ID and values from the line
    std::getline(ss, idString, ';');
    std::getline(ss, valuesString, ']');
    
    // Remove the first 2 characters from valuesString (" [")
    valuesString.erase(0, 2);

    std::vector<int> values;
    std::stringstream valuesStream(valuesString);
    std::string value;
    //std::getline(valuesStream, value, ',');
    while (std::getline(valuesStream, value, ',')) {
      valuesString.erase(0, value.size());
      values.push_back(std::stoi(value));
    }
    
    // Remove the first 4 characters from idString (they are "ID: ")
    idString.erase(0, 4);
    int id = std::stoi(idString);
    
    // Add the ID as the last element of the values vector
    values.push_back(id);
    scalars.push_back(values);
  }
  
  inputFile.close();

}