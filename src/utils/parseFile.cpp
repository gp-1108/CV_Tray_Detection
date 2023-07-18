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
    std::getline(ss, valuesString, '[');
    valuesString.pop_back(); // Remove the trailing ']'
    
    std::vector<int> values;
    std::stringstream valuesStream(valuesString);
    std::string value;
    
    while (std::getline(valuesStream, value, ',')) {
        values.push_back(std::stoi(value));
    }
    
    int id = std::stoi(idString);
    
    // Add the ID as the last element of the values vector
    values.push_back(id);
    
    // Print the vector values
    for (const auto& val : values) {
        std::cout << val << " ";
    }
    
    std::cout << std::endl;
  }
  
  inputFile.close();

}
