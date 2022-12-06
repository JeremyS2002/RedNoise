#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Utils.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

class TextureMap {
public:
	size_t width;
	size_t height;
	std::vector<glm::vec3> pixels;

	TextureMap();
	TextureMap(const std::string &filename);
	friend std::ostream &operator<<(std::ostream &os, const TextureMap &point);
};
