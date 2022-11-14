#pragma once

#include "TexturePoint.h"
#include <iostream>
#include <glm/glm.hpp>

struct CanvasPoint {
	float x{};
	float y{};
	float depth{};
	float brightness{};
	TexturePoint texturePoint{};

	CanvasPoint();
	CanvasPoint(float xPos, float yPos);
	CanvasPoint(float xPos, float yPos, float pointDepth);
	CanvasPoint(float xPos, float yPos, float pointDepth, float pointBrightness);
	CanvasPoint(glm::vec3 pos, glm::vec2 uv);
	friend std::ostream &operator<<(std::ostream &os, const CanvasPoint &point);
};
