#include "TextureMap.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

TextureMap::TextureMap() = default;
TextureMap::TextureMap(const std::string &filename) {
	std::ifstream inputStream(filename, std::ifstream::binary);
	std::string nextLine;
	// Get the "P6" magic number
	std::getline(inputStream, nextLine);
	// Read the width and height line
	std::getline(inputStream, nextLine);
	// Skip over any comment lines !
	while (nextLine.at(0) == '#') std::getline(inputStream, nextLine);
	auto widthAndHeight = split(nextLine, ' ');
	if (widthAndHeight.size() != 2)
		throw std::invalid_argument("Failed to parse width and height line, line was `" + nextLine + "`");

	width = std::stoi(widthAndHeight[0]);
	height = std::stoi(widthAndHeight[1]);
	// Read the max value (which we assume is 255)
	std::getline(inputStream, nextLine);

	// pixels.resize(width * height);
	for (size_t i = 0; i < width * height; i++) {
		int r = inputStream.get();
		int g = inputStream.get();
		int b = inputStream.get();
		// pixels[i] = ((255 << 24) + (red << 16) + (green << 8) + (blue));
		pixels.push_back(glm::vec3(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0));
	}
	inputStream.close();
}

std::ostream &operator<<(std::ostream &os, const TextureMap &map) {
	os << "(" << map.width << " x " << map.height << ")";
	return os;
}
