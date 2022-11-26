#include <DrawingWindow.h>
#include <Utils.h>
#include <fstream>
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <Colour.h>
#include <TextureMap.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <thread>
#include <random>

#define WIDTH 640
#define HEIGHT 480

#define PI 3.14159265

template <typename T>
/// @brief interpolate between from and to n times and return calculated values in a std vector 
/// @tparam T 
/// @param from the starting value
/// @param to the end value
/// @param n the number to interpolate
/// @return the interpolated values as a std::vector<T>
inline std::vector<T> interp(T from, T to, size_t n) {
	std::vector<T> result;

	T delta = (to - from) / T(n - 1);

	T val = from;

	for (size_t i = 0; i < n; i++) {
		result.push_back(val);
		val += delta;
	}

	result.push_back(to);

	return result;
}

struct PixCoord {
	int32_t x;
	int32_t y;
	float depth;
};

struct DeviceCoord {
	glm::vec3 pos;
};

struct WorldCoord {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
};

class Framebuffer {
	public:
		virtual size_t width() = 0;

		virtual size_t height() = 0;

		virtual void putPixelNoDepth(PixCoord p, glm::vec4 color) = 0;

		/// @brief color the pixel to this color applying any blending setup already
		/// @param ndc the coordinates to access
		/// @param color the color to use
		virtual void putPixel(PixCoord p, glm::vec4 color) = 0;
};

class Camera {
	public:
		Camera(float theta) {
			this->pos = float(5.0) * glm::vec3(cos(-theta + (PI/2)), 0.0, sin(-theta + (PI/2)));
			this->f = 2.0;
			this->pitch = 0.0;
			this->yaw = theta;
			this->lastPos = this->pos;
			this->lastPitch = this->pitch;
			this->lastYaw = this->yaw;
		}

		glm::vec3 getPos() {
			return this->pos;
		}

		float getPitch() {
			return this->pitch;
		}

		float getYaw() {
			return this->yaw;
		}

		glm::vec3 getLastPos() {
			return this->lastPos;
		}

		float getLastPitch() {
			return this->lastPitch;
		}

		float getLastYaw() {
			return this->lastYaw;
		}

		void setPos(glm::vec3 newPos) {
			this->lastPos = this->pos;
			this->pos = newPos;
		}

		void setPitch(float newPitch) {
			this->lastPitch = this->pitch;
			this->pitch = newPitch;
		}

		void setYaw(float newYaw) {
			this->lastYaw = this->yaw;
			this->yaw = newYaw;
		}

		void addPos(glm::vec3 deltaPos) {
			this->lastPos = this->pos;
			this->pos += deltaPos;
		}

		void addPitch(float deltaPitch) {
			this->lastPitch = this->pitch;
			this->pitch += deltaPitch;
		}

		void addYaw(float deltaYaw) {
			this->lastYaw = this->yaw;
			this->yaw += deltaYaw;
		}

		float getF() {
			return this->f;
		}
	private:
		float f;
		
		glm::vec3 pos;
		float pitch;
		float yaw;

		glm::vec3 lastPos;
		float lastPitch;
		float lastYaw;
};

struct Material {
	/// @brief base color of the material rgba
	glm::vec4 color;
	/// @brief how much specular components of lighting contribute to the render
	float roughness;
	/// @brief name of the material (to look up faces in draw)
	std::string name;
};

struct FaceIndices {
	/// @brief vertex 0 position index
	uint32_t p0;
	/// @brief vertex 1 position index
	uint32_t p1;
	/// @brief vertex 2 position index
	uint32_t p2;

	bool hasNormals;
	uint32_t n0;
	uint32_t n1;
	uint32_t n2;

	bool hasUvs;
	uint32_t uv0;
	uint32_t uv1;
	uint32_t uv2;

	int mtl;
};

struct Face {
	bool hasNormals;
	bool hasUvs;
	WorldCoord a;
	WorldCoord b;
	WorldCoord c;
	int mtl;
};

class Mesh {
	public:
		std::vector<Face> faces;
		std::vector<Material> materials;

	Mesh(std::string path, glm::mat4 model) {
		std::string buf;
		std::ifstream F(path);

		std::string mtllib;
		// read mtl file name
		while(getline(F, buf)) {
			auto s = split(buf, ' ');
			if (s[0] == "mtllib") {
				mtllib = s[1];
				break;
			}
		}

		std::ifstream MF(mtllib);

		std::vector<Material> materials_buf;
		Material m;
		m.roughness = 1.0;
		bool first_loop = true;
		
		while (getline(MF, buf)) {
			if (buf.size() == 0) {
				continue;
			}

			std::vector<std::string> vals = split(buf, ' ');

			if (vals[0] == "newmtl") {
				if (!first_loop) {
					materials_buf.push_back(m);
				} else {
					first_loop = false;
				}
				m.name = vals[1];
			} else if (vals[0] == "Kd") {
				float r = std::stof(vals[1]);
				float g = std::stof(vals[2]);
				float b = std::stof(vals[3]);
				float a = 1.0;
				if (vals.size() >= 5) {
					a = std::stof(vals[4]);
				}
				m.color = glm::vec4(r, g, b, a);
			}
		}

		materials_buf.push_back(m);

		MF.close();

		this->materials = materials_buf;

		std::vector<glm::vec3> positions_buf;
		std::vector<glm::vec3> normals_buf;	
		std::vector<glm::vec2> uvs_buf;
		std::vector<FaceIndices> faces_buf;
		std::string current_mtl;
		bool mtl_set = false;
		std::unordered_map<std::string, int> material_name_map;

		for (int i = 0; i < materials.size(); i++) {
			material_name_map[materials[i].name] = i;
		}

		while (getline(F, buf)) {
			// std::cout << buf << std::endl;
			if (buf.size() == 0) {
				continue;
			}

			std::vector<std::string> vals = split(buf, ' ');

			if (vals[0] == "v") {
				float x = std::stof(vals[1]);
				float y = std::stof(vals[2]);
				float z = std::stof(vals[3]);
				glm::vec3 pos = glm::vec3(x, y, z);	
				positions_buf.push_back(pos);
			} else if (vals[0] == "vn") {
				float x = std::stof(vals[1]);
				float y = std::stof(vals[2]);
				float z = std::stof(vals[3]);
				glm::vec3 normal = glm::vec3(x, y, z);
				normals_buf.push_back(normal);
			} else if (vals[0] == "vt") {
				float x = std::stof(vals[1]);
				float y = std::stof(vals[2]);
				glm::vec2 uv = glm::vec2(x, y);
				uvs_buf.push_back(uv);
			} else if (vals[0] == "f") {
				auto vals1split = split(vals[1], '/');
				auto vals2split = split(vals[2], '/');
				auto vals3split = split(vals[3], '/');

				FaceIndices f;

				f.p0 = std::stoi(vals1split[0]) - 1;
				f.p1 = std::stoi(vals2split[0]) - 1;
				f.p2 = std::stoi(vals3split[0]) - 1;

				if (vals1split.size() > 1 && vals2split.size() > 1 && vals3split.size() > 1) {
					if (!vals1split[1].empty() && !vals1split[1].empty() && !vals1split[1].empty()) {
						f.hasUvs = true;
						f.uv0 = std::stoi(vals1split[1]) - 1;
						f.uv1 = std::stoi(vals2split[1]) - 1;
						f.uv2 = std::stoi(vals3split[1]) - 1;
					} else {
						f.hasUvs = false;
					}
				} else {
					f.hasUvs = false;
				}

				if (vals1split.size() > 2 && vals2split.size() > 2 && vals3split.size() > 2) {
					if (!vals1split[2].empty() && !vals1split[2].empty() && !vals1split[2].empty()) {
						f.hasNormals = true;
						f.n0 = std::stoi(vals1split[2]) - 1;
						f.n1 = std::stoi(vals2split[2]) - 1;
						f.n2 = std::stoi(vals3split[2]) - 1;
					} else {
						f.hasNormals = false;
					}
				} else {
					f.hasNormals = false;
				}

				if (mtl_set) {
					f.mtl = material_name_map[current_mtl];
				} else {
					f.mtl = -1;
				}

				faces_buf.push_back(f);
			} else if (vals[0] == "usemtl") {
				// change the current material being used
				mtl_set = true;
				current_mtl = vals[1];
			}
		}

		F.close();

		this->faces = std::vector<Face>(faces.size());
		for (auto face = faces_buf.begin(); face != faces_buf.end(); face++) {
			WorldCoord a;
			a.pos = positions_buf[face->p0];
			a.pos = glm::vec3(model * glm::vec4(a.pos.x, a.pos.y, a.pos.z, 1.0));
			WorldCoord b;
			b.pos = positions_buf[face->p1];
			b.pos = glm::vec3(model * glm::vec4(b.pos.x, b.pos.y, b.pos.z, 1.0));
			WorldCoord c;
			c.pos = positions_buf[face->p2];
			c.pos = glm::vec3(model * glm::vec4(c.pos.x, c.pos.y, c.pos.z, 1.0));

			if (face->hasUvs) {
				a.uv = uvs_buf[face->uv0];
				b.uv = uvs_buf[face->uv1];
				c.uv = uvs_buf[face->uv2];
			}

			if (face->hasNormals) {
				glm::mat3 normalModel = glm::transpose(glm::inverse(glm::mat3(model)));
				a.normal = normalModel * normals_buf[face->n0];
				b.normal = normalModel * normals_buf[face->n1];
				c.normal = normalModel * normals_buf[face->n2];						
			}

			Face f;
			f.hasNormals = face->hasNormals;
			f.hasUvs = face->hasUvs;
			f.a = a;
			f.b = b;
			f.c = c;
			f.mtl = face->mtl;
			this->faces.push_back(f);
		}
	}
};

/// @brief convert normalized device coordinates into screen coordinates
/// @param ndc the coordinates in normalized device coordinate system (-1.0 to 1.0)
/// @param width the width of the framebuffer in pixels
/// @param height the height of the framebuffer in pixels
/// @return the coordinates in pixel positions x: (0, width) and y: (0, height)
inline PixCoord device2pix(DeviceCoord ndc, uint width, uint height) {	
	auto half = glm::vec2(0.5);
	auto t = (half * glm::vec2(ndc.pos) + half) * glm::vec2(float(width - 1), float(height - 1));
	PixCoord p;
	p.x = round(t.x);
	p.y = round(t.y);
	p.depth = ndc.pos.z;
	return p;
}

inline DeviceCoord world2device(WorldCoord world, Camera *cam) {
	glm::vec3 t = world.pos - cam->getPos();
	glm::mat3 pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->getPitch()), -sin(cam->getPitch())),
		glm::vec3(0.0, sin(cam->getPitch()), cos(cam->getPitch()))
	);
	glm::mat3 yaw_mat = glm::mat3(
		glm::vec3(cos(cam->getYaw()), 0.0, sin(cam->getYaw())),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(cam->getYaw()), 0.0, cos(cam->getYaw()))
	);
	DeviceCoord device;
	t = yaw_mat * t;
	t = pitch_mat * t;
	device.pos.x = -cam->getF() * (t.x / t.z);
	device.pos.y = cam->getF() * (t.y / t.z);
	device.pos.z = -t.z;
	return device;
}

/// @brief draw a line in pixel coordinates
/// @param from start position of the line
/// @param to end position of the line
/// @param mtl material to draw the line in
/// @param f framebuffer to draw to
inline void drawPixLine(PixCoord from, PixCoord to, Material mtl, Framebuffer *f) {
	float x_diff = float(to.x) - float(from.x);
	float y_diff = float(to.y) - float(from.y);
	float depth_diff = std::abs(to.depth - from.depth);
	uint32_t n = uint32_t(std::max(std::abs(x_diff), std::abs(y_diff)));

	float x_step = x_diff / n;
	float y_step = y_diff / n;
	float depth_step = depth_diff / n;

	PixCoord draw;
	for (uint32_t i = 0; i < n; i++) {
		draw.x = from.x + uint32_t(x_step * i);
		draw.y = from.y + uint32_t(y_step * i);
		draw.depth = from.depth + depth_step * i;
		f->putPixelNoDepth(draw, mtl.color);
	}
}

/// @brief draw a line to the framebuffer
/// @param from from coordinate in ndc
/// @param to to coordinate in ndc
/// @param mtl material for the line
/// @param f framebuffer to draw to
inline void drawDeviceLine(DeviceCoord from, DeviceCoord to, Material mtl, Framebuffer *f) {
	auto width = f->width();
	auto height = f->height();

	PixCoord pix_from = device2pix(from, width, height);
	PixCoord pix_to = device2pix(to, width, height);	

	drawPixLine(pix_from, pix_to, mtl, f);
}

/// @brief draw a line to the framebuffer
/// @param from from coordinate in world coordinates
/// @param to to coordinate in world coordinates
/// @param mtl material for the line
/// @param f framebuffer to draw to
/// @param c camera to transform points by
/// @param model model matrix to transform points by
inline void drawWorldLine(WorldCoord from, WorldCoord to, Material mtl, Framebuffer *f, Camera *cam, glm::mat4 model) {
	DeviceCoord device_from = world2device(from, cam);
	DeviceCoord device_to = world2device(to, cam);
	drawDeviceLine(device_from, device_to, mtl, f);
}

/// @brief draw a wireframe triangle from pixel coordinates
/// @param a first vertex
/// @param b second vertex
/// @param c third vertex
/// @param mtl material to draw the triangle in
/// @param f framebuffer to draw to
inline void drawPixTriangleWireFrame(PixCoord a, PixCoord b, PixCoord c, Material mtl, Framebuffer *f) {
	// std::cout << "Pix triangle wireframe" << std::endl;
	// std::cout << a.x << " " << a.y << " " << a.depth << std::endl;
	// std::cout << b.x << " " << b.y << " " << b.depth << std::endl;
	// std::cout << c.x << " " << c.y << " " << c.depth << std::endl;
	// std::cout << std::endl;

	drawPixLine(a, b, mtl, f);
	drawPixLine(b, c, mtl, f);
	drawPixLine(c, a, mtl, f);
}

/// @brief draw a wireframe triangle from device coordinates
/// @param a first vertex position in ndc
/// @param b second vertex position in ndc
/// @param c third vertex position in ndc
/// @param mtl material for the triangle lines
/// @param f framebuffer to draw to
inline void drawDeviceTriangleWireFrame(DeviceCoord a, DeviceCoord b, DeviceCoord c, Material mtl, Framebuffer *f) {
	// std::cout << "Device triangle wireframe" << std::endl;
	// std::cout << a.pos.x << " " << a.pos.y << " " << a.pos.z << std::endl;
	// std::cout << b.pos.x << " " << b.pos.y << " " << b.pos.z << std::endl;
	// std::cout << c.pos.x << " " << c.pos.y << " " << c.pos.z << std::endl;
	// std::cout << std::endl;

	auto width = f->width();
	auto height = f->height();

	PixCoord pix_a = device2pix(a, width, height);
	PixCoord pix_b = device2pix(b, width, height);
	PixCoord pix_c = device2pix(c, width, height);

	drawPixTriangleWireFrame(pix_a, pix_b, pix_c, mtl, f);
}

/// @brief draw a wireframe triangle from world coordinates
/// @param a the first vertex position in world space
/// @param b the second vertex position in world space
/// @param c the third vertex position in world space
/// @param mtl the material for the triangle lines
/// @param f the framebuffer to draw to
/// @param c camera to transofrm points by
/// @param model model matrix to transform points by
inline void drawWorldTriangleWireframe(WorldCoord a, WorldCoord b, WorldCoord c, Material mtl, Framebuffer *f, Camera *cam, glm::mat4 model) {
	auto device_a = world2device(a, cam);
	auto device_b = world2device(b, cam);
	auto device_c = world2device(c, cam);
	
	drawDeviceTriangleWireFrame(device_a, device_b, device_c, mtl, f);
}

inline void __internalflatPixTriangle(PixCoord a, PixCoord b, PixCoord c, Material mtl, Framebuffer *f) {
	// std::cout << a.x << " " << a.y << std::endl;
	// std::cout << b.x << " " << b.y << std::endl;
	// std::cout << c.x << " " << c.y << std::endl;
	assert(b.y == c.y);
	assert(a.y != b.y);
	assert(a.y != c.y);

	bool top;
	if (a.y < b.y) {
		top = false;
	} else {
		top = true;
	}

	// glm::vec2 ta = glm::vec2(triangle.v0().texturePoint.x, triangle.v0().texturePoint.y);
	// glm::vec2 tb = glm::vec2(triangle.v1().texturePoint.x, triangle.v1().texturePoint.y);
	// glm::vec2 tc = glm::vec2(triangle.v2().texturePoint.x, triangle.v2().texturePoint.y);

	if (b.x > c.x) {
		std::swap(b, c);
		// std::swap(tb, tc);
	}

	glm::vec3 ab = glm::vec3(b.x, b.y, b.depth) - glm::vec3(a.x, a.y, a.depth);
	glm::vec3 ac = glm::vec3(c.x, c.y, c.depth) - glm::vec3(a.x, a.y, a.depth);

	glm::vec3 unit_ab = glm::normalize(ab);
	glm::vec3 unit_ac = glm::normalize(ac);

	uint32_t diff_y;
	if (top) {
		diff_y = a.y - b.y;
	} else {
		diff_y = b.y - a.y;
	}

	float tab = 0.0;
	float tac = 0.0;
	float step_ab = glm::length(ab) / diff_y;
	float step_ac = glm::length(ac) / diff_y;

	if (top) {
		step_ab *= -1.0;
		step_ac *= -1.0;
	}
	// auto tex_ab = interp(ta, tb, sb.y - sa.y);
	// auto tex_ac = interp(ta, tc, sc.y - sa.y);

	int32_t start_y;
	int32_t end_y;

	glm::vec3 start_ab;
	glm::vec3 start_ac;

	if (top) {
		start_y = b.y;
		end_y = a.y;
		start_ab = glm::vec3(b.x, b.y, b.depth);
		start_ac = glm::vec3(c.x, c.y, c.depth);
	} else {
		start_y = a.y;
		end_y = b.y;
		start_ab = glm::vec3(a.x, a.y, a.depth);
		start_ac = glm::vec3(a.x, a.y, a.depth);
	}

	PixCoord p;
	for (int32_t y = start_y; y <= end_y; y++) {
		p.y = y;

		glm::vec3 l = start_ab + tab * unit_ab;
		glm::vec3 r = start_ac + tac * unit_ac;
		
		float depth = l.z;
		float ddepth = (r.z - l.z) / (int32_t(r.x) - int32_t(l.x));

		for (int32_t x = int32_t(l.x); x <= int32_t(r.x); x++) {
			p.x = x;
			p.depth = depth;
			
			f->putPixel(p, mtl.color);

			depth += ddepth;
		}

		tab += step_ab;
		tac += step_ac;
	}
}

// check depth interp on line
// depth -inf

/// @brief draw a triangle to the framebuffer
/// @param a first vertex in ndc
/// @param b second vertex in ndc
/// @param c third vertex in ndc
/// @param mtl material to use for the triangle
/// @param f the framebuffer to draw to
void drawPixTriangle(PixCoord a, PixCoord b, PixCoord c, Material mtl, Framebuffer *f) {
	// std::cout << "Pix triangle" << std::endl;
	// std::cout << a.x << " " << a.y << " " << a.depth << std::endl;
	// std::cout << b.x << " " << b.y << " " << b.depth << std::endl;
	// std::cout << c.x << " " << c.y << " " << c.depth << std::endl;
	// std::cout << std::endl;

	if (a.y == b.y) {
		// a and b are already on the same height
		// swap a and c so that c and c make the flat edge
		std::swap(a, c);
		
		if (b.x > c.x) {
			// put b to the left of c
			std::swap(b, c);
		}

		__internalflatPixTriangle(a, b, c, mtl, f);
	} else if (a.y == c.y) {
		// a and c are already on the same height
		// swap a and b so b and c make the flat edge
		std::swap(a, b);
		
		if (b.x > c.x) {
			// make b left of c
			std::swap(b, c);
		}

		__internalflatPixTriangle(a, b, c, mtl, f);
	} else if (b.y == c.y) {
		// b and c are already on the same edge

		if (b.x > c.x) {
			// make b left of c
			std::swap(b, c);
		}

		__internalflatPixTriangle(a, b, c, mtl, f);
	} else {
		// none of the vertices are on the same height
		// need to split the triange into two

		// order the vertiecs by height
		if (b.y < a.y) {
			std::swap(a, b);
		}

		if (c.y < a.y) {
			std::swap(c, a);
		}

		if (c.y < b.y) {
			std::swap(c, b);
		}

		PixCoord d;
		float t = (float(b.y) - float(a.y)) / (float(c.y) - float(a.y));
		d.x = int32_t(float(a.x) + t * (float(c.x) - float(a.x)));
		d.y = b.y;
		d.depth = float(a.depth) + t * (float(c.depth) - float(a.depth));

		// t acts as a percentage between a and c so can interpolate with t
		// glm::vec2 ta = glm::vec2(a.texturePoint.x, a.texturePoint.y);
		// glm::vec2 tc = glm::vec2(c.texturePoint.x, c.texturePoint.y);
		// glm::vec2 td = (float(1.0) - t) * tc + t * ta;

		__internalflatPixTriangle(a, b, d, mtl, f);

		__internalflatPixTriangle(c, b, d, mtl, f);
	}

	// this->triangle(a, b, c, glm::vec3(1.0, 1.0, 1.0));
}

/// @brief draw a triangle into the framebuffer
/// @param a first vertex in world space coordinates
/// @param b second vertex in world space coordinates
/// @param c third vertex in world space coordinates
/// @param mtl material to use for the triangle
/// @param f framebuffer to draw to
/// @param cam camera to transform vertices by
/// @param model model to transform vertices by
inline void drawDeviceTriangle(DeviceCoord a, DeviceCoord b, DeviceCoord c, Material mtl, Framebuffer *f) {
	// std::cout << "Device triangle" << std::endl;
	// std::cout << a.pos.x << " " << a.pos.y << " " << a.pos.z << std::endl;
	// std::cout << b.pos.x << " " << b.pos.y << " " << b.pos.z << std::endl;
	// std::cout << c.pos.x << " " << c.pos.y << " " << c.pos.z << std::endl;
	// std::cout << std::endl;

	uint32_t width = f->width();
	uint32_t height = f->height();

	PixCoord pix_a = device2pix(a, width, height);
	PixCoord pix_b = device2pix(b, width, height);
	PixCoord pix_c = device2pix(c, width, height);

	drawPixTriangle(pix_a, pix_b, pix_c, mtl, f);
}

/// @brief 
/// @param a 
/// @param b 
/// @param c 
/// @param mtl 
/// @param f 
/// @param cam 
/// @param model 
inline void drawWorldTriangle(WorldCoord a, WorldCoord b, WorldCoord c, Material mtl, Framebuffer *f, Camera *cam) {
	DeviceCoord device_a = world2device(a, cam);
	DeviceCoord device_b = world2device(b, cam);
	DeviceCoord device_c = world2device(c, cam);

	drawDeviceTriangle(device_a, device_b, device_c, mtl, f);
}


/// @brief draw a mesh into the framebuffer
/// @param m the mesh to draw
/// @param c the camera to look through
/// @param f the framebuffer to draw into
void drawMesh(Mesh *m, Camera *cam, Framebuffer *f) {
	// Material white;
	// white.color = glm::vec4(1.0);
	// white.name = std::string("white");
	// for (auto mtl = m->materials.begin(); mtl != m->materials.end(); mtl++) {
	// 	if (m->draw.find(mtl->name) != m->draw.end()) {
	// 		auto faces = m->draw.at(mtl->name);
	// 		for (auto face_iter = faces.begin(); face_iter != faces.end(); face_iter++) {
	// 			Face face = m->faces[*face_iter];

	// 			WorldCoord pos0 = m->positions[face.v0];
	// 			WorldCoord pos1 = m->positions[face.v1];
	// 			WorldCoord pos2 = m->positions[face.v2];

	// 			drawWorldTriangle(pos0, pos1, pos2, *mtl, f, cam, m->model);
	// 			// drawWorldTriangleWireframe(pos0, pos1, pos2, white, f, cam, m.model);
	// 		}
	// 	}
	// }
	for (auto face = m->faces.begin(); face != m->faces.end(); face++) {
		drawWorldTriangle(face->a, face->b, face->c, m->materials[face->mtl], f, cam);
	}
}

/// @brief Manages window and basic graphics capabilities
class MyWindow : public Framebuffer {
	public:
		DrawingWindow window;
		bool shouldClose;
		std::vector<float> depth; 

		size_t width() {
			return this->window.width;
		}

		size_t height() {
			return this->window.height;
		}

		void putPixelNoDepth(PixCoord p, glm::vec4 color) {
			if (p.x < 0 || p.x >= this->window.width) {
				return;
			}

			if (p.y < 0 || p.y >= this->window.height) {
				return;
			}

			uint8_t r = int(color.x * 255.0);
			uint8_t g = int(color.y * 255.0);
			uint8_t b = int(color.z * 255.0);
			uint32_t packed = (255 << 24) + (int(r) << 16) + (int(g) << 8) + int(b);

			this->window.setPixelColour(p.x, p.y, packed);
		}

		void putPixel(PixCoord p, glm::vec4 color) {
			if (p.x < 0 || p.x >= this->window.width) {
				return;
			}

			if (p.y < 0 || p.y >= this->window.height) {
				return;
			}

			float current_depth = this->depth[p.x + p.y * this->width()];
			if((1.0 / p.depth) >= current_depth) {
				this->depth[p.x + p.y * this->width()] = 1.0 / p.depth;

				uint8_t r = int(std::fmin(std::fmax(color.x, 0.0), 1.0) * 255.0);
				uint8_t g = int(std::fmin(std::fmax(color.y, 0.0), 1.0) * 255.0);
				uint8_t b = int(std::fmin(std::fmax(color.z, 0.0), 1.0) * 255.0);
				uint32_t packed = (255 << 24) + (int(r) << 16) + (int(g) << 8) + int(b);

				this->window.setPixelColour(p.x, p.y, packed);
			}
		}

	MyWindow(uint width, uint height, bool fullscreen) {
		this->window = DrawingWindow(width, height, fullscreen);
		this->shouldClose = false;
		this->depth = std::vector<float>(width * height, -10000.0);
		// this->texture = TextureMap("texture.ppm");
	}

	inline void clear() {
		this->window.clearPixels();
		this->depth = std::vector<float>(this->width() * this->height(), -1000.0);
	}
};

struct Ray {
	glm::vec3 src;
	glm::vec3 dir;
};

struct Intersection {
	bool valid;
	float t;
	float u;
	float v;
};

inline Intersection rayTriangleIntersection(Ray ray, WorldCoord a, WorldCoord b, WorldCoord c) {
	// std::cout << "ray src: " << ray.src.x << " " << ray.src.y << " " << ray.src.z << std::endl;
	// std::cout << "ray dir: " << ray.dir.x << " " << ray.dir.y << " " << ray.dir.z << std::endl;

	// std::cout << "a pos  : " << a.pos.x << " " << a.pos.y << " " << a.pos.z << std::endl;
	// std::cout << "b pos  : " << b.pos.x << " " << b.pos.y << " " << b.pos.z << std::endl;
	// std::cout << "c pos  : " << c.pos.x << " " << c.pos.y << " " << c.pos.z << std::endl;
	
	glm::vec3 e0 = b.pos - a.pos;
	glm::vec3 e1 = c.pos - a.pos;
	glm::vec3 SPVector = ray.src - a.pos;
	glm::mat3 DEMatrix(-ray.dir, e0, e1);
	if (glm::determinant(DEMatrix) == 0.0) {
		// ray parallel to plane
		Intersection i;
		i.valid = false;
		return i;
	}
	glm::vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;
	Intersection i;
	i.valid = true;
	i.t = possibleSolution.x;
	i.u = possibleSolution.y;
	i.v = possibleSolution.z;

	if (i.u < 0.0 || i.u > 1.0) {
		i.valid = false;
	}

	if (i.v < 0.0 || i.v > 1.0) {
		i.valid = false;
	}

	if (i.u + i.v > 1.0) {
		i.valid = false;
	}

	if (i.t <= 0.0) {
		i.valid = false;
	}

	return i;
}

struct IntersectionInfo {
	float t;
	float u;
	float v;
	Face face;
};

struct RayResult {
	bool valid;
	IntersectionInfo intersection;
};

inline RayResult traceRay(Ray ray, Mesh *m) {
	// std::vector<IntersectionInfo> intersections;// = std::vector<IntersectionInfo>(m->faces.size());
	bool intersectionFound = false;
	IntersectionInfo info;

	for (auto face = m->faces.begin(); face != m->faces.end(); face++) {
		Intersection i = rayTriangleIntersection(ray, face->a, face->b, face->c);
		if (i.valid) {
			// std::cout << face->mtl.name << std::endl;
			if (intersectionFound) {
				if (i.t < info.t) {
					info.t = i.t;
					info.u = i.u;
					info.v = i.v;
					info.face = *face;
				}
			} else {
				info.t = i.t;
				info.u = i.u;
				info.v = i.v;
				info.face = *face;
				intersectionFound = true;
			}
		}
	}

	if (intersectionFound) {
		RayResult r;
		r.valid = true;
		r.intersection = info;
		return r;
	} else {
		RayResult r;
		r.valid = false;
		return r;
	}
}

struct PointLight {
	glm::vec3 pos;
	float radius;
	float falloff;
	glm::vec3 color;
	std::vector<float> shadowMapA;
	std::vector<float> shadowMapB;
};

class Environment {
	public:
		std::vector<PointLight> pointLights;
		float ambient;

		Environment(std::vector<PointLight> pointLights, float ambient) {
			this->pointLights = pointLights;
			this->ambient = ambient;
		}
};

template <typename T>
inline T interpUV(T a, T b, T c, float u, float v) {
	T d = a + (b - a) + (c - a);
	return a*(float(1.0)-u)*(float(1.0)-v) + b*u*(float(1.0)-v) + c*(float(1.0)-u)*v + d*u*v;
}

inline glm::vec3 fresnelSchlick(float cos_theta, glm::vec3 f0) {
	return f0 + (glm::vec3(1.0) - f0) * powf32(1.0 - cos_theta, 5.0);
}

inline glm::vec3 fresnelSchlickRoughness(float cos_theta, glm::vec3 f0, float roughness) {
	return f0 + (glm::max(glm::vec3(1.0 - roughness), f0), -f0) * powf32(std::fmax(1.0 - cos_theta, 0.0), 5.0);
}

inline float distrubitionGGX(glm::vec3 n, glm::vec3 h, float roughness) {
	float a = roughness * roughness;
	float a2 = a * a;
	float n_dot_h = std::fmax(glm::dot(n, h), 0.0);
	float denom = (n_dot_h * n_dot_h * (a2 - 1.0) + 1.0);
	return a2 / (PI * denom * denom);
}

inline float geometry_schlickGGX(float n_dot_v, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

inline float geometry_smith(glm::vec3 n, glm::vec3 v, glm::vec3 l, float roughness) {
	float n_dot_v = std::fmax(glm::dot(n, v), 0.0);
	float n_dot_l = std::fmax(glm::dot(n, l), 0.0);
	float ggx2 = geometry_schlickGGX(n_dot_v, roughness);
	float ggx1 = geometry_schlickGGX(n_dot_l, roughness);

	return ggx1 * ggx2;
}

inline glm::vec3 point_light_calc(
	PointLight light,
	glm::vec3 view_pos,
	glm::vec3 world_pos,
	glm::vec3 normal,
	glm::vec3 albedo,
	float roughness,
	float metallic
) {
	glm::vec3 to_light = light.pos - world_pos;
	glm::vec3 to_light_unit = glm::normalize(to_light);
	glm::vec3 view = glm::normalize(view_pos - world_pos);

	glm::vec3 halfway = glm::normalize(view + to_light_unit);

	float distance2 = glm::dot(to_light, to_light);
	float attenuation = 1.0 / (0.0001 + light.falloff * distance2);
	glm::vec3 radiance = light.color * attenuation;

	glm::vec3 f0 = glm::vec3(0.04);
	f0 = glm::mix(f0, albedo, metallic);
	glm::vec3 f = fresnelSchlick(std::fmax(glm::dot(halfway, view), 0.0), f0);

	float ndf = distrubitionGGX(normal, halfway, roughness);
	float g = geometry_smith(normal, view, to_light_unit, roughness);

	glm::vec3 numerator = ndf * g * f;
	float denominator = 4.0 * std::fmax(glm::dot(normal, view), 0.0) * std::fmax(glm::dot(normal, to_light_unit), 0.0);
	glm::vec3 specular = numerator / std::fmax(denominator, float(0.001));

	glm::vec3 ks = f;
	glm::vec3 kd = glm::vec3(1.0) - ks;
	kd *= 1.0 - metallic;

	float n_dot_l = std::fmax(glm::dot(normal, to_light_unit), 0.0);

	return (kd * albedo / float(PI) + specular) * radiance * n_dot_l;
}

class GeometryBuffer {
	public:
		size_t width;
		size_t height;
		std::vector<glm::vec3> world_pos;
		std::vector<glm::vec3> normal;
		std::vector<glm::vec2> uv;
		std::vector<int> material;
		std::vector<float> depth;
		std::vector<glm::vec2> motion;

		GeometryBuffer(size_t width, size_t height) {
			this->width = width;
			this->height = height;
			this->world_pos = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->normal = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->uv = std::vector<glm::vec2>(width * height, glm::vec2(0.0));
			this->depth = std::vector<float>(this->width * this->height, 0.0);
			this->material = std::vector<int>(this->width * this->height, -1);
			this->motion = std::vector<glm::vec2>(this->width * this->height, glm::vec2(0.0));
		}

		void clear() {
			this->world_pos = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->normal = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->uv = std::vector<glm::vec2>(width * height, glm::vec2(0.0));
			this->depth = std::vector<float>(this->width * this->height, 0.0);
			this->material = std::vector<int>(this->width * this->height, -1);			
			this->motion = std::vector<glm::vec2>(this->width * this->height, glm::vec2(0.0));
		}

		inline size_t index(size_t x, size_t y) {
			if (x < 0 || x >= this->width) {
				return -1;
			}

			if (y < 0 || y >= this->height) {
				return -1;
			}

			return x + y * this->width;
		}
};

inline void traceMeshBounded(Mesh *m, Camera *cam, GeometryBuffer *g, size_t sx, size_t sy, size_t w, size_t h) {
	glm::mat3 pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->getPitch()), -sin(cam->getPitch())),
		glm::vec3(0.0, sin(cam->getPitch()), cos(cam->getPitch()))
	);
	glm::mat3 yaw_mat = glm::mat3(
		glm::vec3(cos(-cam->getYaw()), 0.0, sin(-cam->getYaw())),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(-cam->getYaw()), 0.0, cos(-cam->getYaw()))
	);
	glm::mat3 mat = pitch_mat * yaw_mat;

	glm::mat3 last_pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->getLastPitch()), -sin(cam->getLastPitch())),
		glm::vec3(0.0, sin(cam->getLastPitch()), cos(cam->getLastPitch()))
	);
	glm::mat3 last_yaw_mat = glm::mat3(
		glm::vec3(cos(-cam->getLastYaw()), 0.0, sin(-cam->getLastYaw())),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(-cam->getLastYaw()), 0.0, cos(-cam->getLastYaw()))
	);
	glm::mat3 last_mat = last_pitch_mat * last_yaw_mat;

	for (size_t y = sy; y < sy + h; y++) {
		for (size_t x = sx; x < sx + w; x++) {
			Ray ray;
			ray.src = cam->getPos();
			ray.dir = glm::normalize(mat * glm::vec3(
					((2 * float(x)) / g->width - 1) * (float(g->width) / float(g->height)),
					-((2 * float(y)) / g->height - 1),
					-cam->getF()));
			
			RayResult r = traceRay(ray, m);
			if (r.valid) {
				IntersectionInfo info = r.intersection;

				glm::vec3 world_pos = ray.src + ray.dir * info.t;

				Face face = info.face;
				glm::vec3 e1 = face.b.pos - face.a.pos;
				glm::vec3 e2 = face.c.pos - face.a.pos;
				glm::vec3 normal;
				if (face.hasNormals) {
					normal = glm::normalize(interpUV(face.a.normal, face.b.normal, face.c.normal, info.u, info.v));
				} else {
					normal = glm::normalize(glm::cross(e1, e2));
				}

				glm::vec2 uv;
				if (face.hasUvs) {
					uv = interpUV(face.a.uv, face.b.uv, face.c.uv, info.u, info.v);
				} else {
					uv = glm::vec2(0.0);
				}

				glm::vec3 last_device_pos = last_mat * (world_pos - cam->getPos());
				last_device_pos.x = -cam->getF() * (last_device_pos.x / last_device_pos.z);
				last_device_pos.y = cam->getF() * (last_device_pos.y / last_device_pos.z);
				last_device_pos.z = -last_device_pos.z;

				glm::vec2 half = glm::vec2(0.5);
				glm::vec2 screen_pos = glm::vec2(float(x), float(y));
				glm::vec2 last_screen_pos = (half * glm::vec2(last_device_pos) + half) * glm::vec2(float(g->width - 1), float(g->height - 1));
				glm::vec2 motion = screen_pos - last_screen_pos;

				size_t index = g->index(x, y);
				g->world_pos[index] = world_pos;
				g->normal[index] = normal;
				g->uv[index] = uv;
				g->material[index] = face.mtl;
				g->depth[index] = info.t;
				g->motion[index] = motion;
			}
		}
	}
}

inline void traceMeshThreaded(Mesh *m, Camera *cam, GeometryBuffer *g, size_t numThreads) {
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(traceMeshBounded, m, cam, g, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

inline void traceMesh(Mesh *m, Camera *cam, GeometryBuffer *g) {
	traceMeshBounded(m, cam, g, 0, 0, g->width, g->height);
}

float nextRand() {
	// https://stackoverflow.com/questions/686353/random-float-number-generation
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// https://medium.com/@alexander.wester/ray-tracing-soft-shadows-in-real-time-a53b836d123b
glm::vec3 getConeSample(glm::vec3 direction, float coneAngle) {
	float cosAngle = cos(coneAngle);
	float z = nextRand() * (1.0 - cosAngle) + cosAngle;
	float phi = nextRand() * 2.0 * PI;

	float x = sqrt(1.0 - z * z) * cos(phi);
	float y = sqrt(1.0 - z * z) * sin(phi);
	glm::vec3 north = glm::vec3(0.0, 0.0, 1.0);

	glm::vec3 axis = glm::normalize(glm::cross(north, glm::normalize(direction)));
	float angle = acos(glm::dot(glm::normalize(direction), north));

	glm::mat3 R = glm::mat3(glm::rotate(glm::mat4(), angle, axis));
	return R * glm::vec3(x, y, z);
}

// https://medium.com/@alexander.wester/ray-tracing-soft-shadows-in-real-time-a53b836d123b
inline void shadowBounded(Mesh *m, GeometryBuffer *g, Environment *env, float alpha, size_t sx, size_t sy, size_t w, size_t h) {
	for (size_t y = sy; y < sy+h; y++) {
		for (size_t x = sx; x < sx+w; x++) {
			size_t index = g->index(x, y);
			int mtl_idx = g->material[index];
			if (mtl_idx == -1) {
				continue;
			}

			glm::vec3 world_pos = g->world_pos[index];
			glm::vec3 normal = g->normal[index];

			for (auto &light : env->pointLights) {
				glm::vec3 toLight = glm::normalize(light.pos - world_pos);
				glm::vec3 perpL = glm::cross(toLight, glm::vec3(0.0, 1.0, 0.0));
				if (perpL.x == 0.0 && perpL.y == 0.0 && perpL.z == 0.0) {
					perpL.x = 1.0;
				}

				glm::vec3 toLightEdge = glm::normalize((light.pos + perpL * light.radius) - world_pos);
				float angle = std::acos(glm::dot(toLight, toLightEdge)) * 2.0;
				
				float density = 0.0;
				int samples = 1;
				for (int i = 0; i < samples; i++) {
					Ray ray;
					ray.dir = getConeSample(toLight, angle);
					ray.src = world_pos + float(0.0001) * normal;

					RayResult res = traceRay(ray, m);

					if (res.valid) {
						if (res.intersection.t < glm::length(light.pos - world_pos)) {
							density += 1.0;
						}
					}
				}
			
				glm::vec2 motion = g->motion[index];
				std::cout << motion.x << " " << motion.y << std::endl;
				glm::vec2 prev_coord = glm::vec2(x, y) - motion;
				size_t prev_index = g->index(int(prev_coord.x), int(prev_coord.y));
				float prev_density = light.shadowMapB[prev_index];
				light.shadowMapA[index] = alpha * (density / float(samples)) + (float(1.0) - alpha) * prev_density;
			}
		}
	}
}

inline void shadowThreaded(Mesh *m, GeometryBuffer *g, Environment *env, float alpha, size_t numThreads) {
	for (auto &light : env->pointLights) {
		std::swap(light.shadowMapA, light.shadowMapB);
		light.shadowMapA = std::vector<float>(WIDTH * HEIGHT, 0.0);
	}

	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(shadowBounded, m, g, env, alpha, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

inline void shadow(Mesh *m, GeometryBuffer *g, Environment *env, float alpha) {
	for (auto &light : env->pointLights) {
		std::swap(light.shadowMapA, light.shadowMapB);
		light.shadowMapA = std::vector<float>(WIDTH * HEIGHT, 0.0);
	}

	shadowBounded(m, g, env, alpha, 0, 0, g->width, g->height);
}


inline void lightingBounded(Mesh *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env, size_t sx, size_t sy, size_t w, size_t h) {
	for (size_t y = sy; y < sy+h; y++) {
		for (size_t x = sx; x < sx+w; x++) {
			size_t index = g->index(x, y);
			int mtl_idx = g->material[index];
			if (mtl_idx == -1) {
				continue;
			}

			glm::vec3 world_pos = g->world_pos[index];
			glm::vec3 normal = g->normal[index];
			float depth = g->depth[index];
			Material mtl = m->materials[mtl_idx];

			// TODO load from material
			float roughness = 0.9;
			float metallic = 0.0;
			glm::vec3 albedo = glm::vec3(mtl.color);

			PixCoord p;
			p.x = x;
			p.y = y;
			p.depth = depth;
			
			glm::vec3 result = glm::vec3(0.0); 
			result += albedo * env->ambient;

			for (auto &light : env->pointLights) {
				glm::vec3 light_color = point_light_calc(light, cam->getPos(), world_pos, normal, albedo, roughness, metallic);
				float shadow = light.shadowMapA[index];
				result += (float(1.0) - shadow) * light_color;
			}

			f->putPixel(p, glm::vec4(result.x, result.y, result.z, 1.0));
		}
	}
}

inline void lightingThreaded(Mesh *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env, size_t numThreads) {
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(lightingBounded, m, g, f, cam, env, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

inline void lighting(Mesh *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env) {
	lightingBounded(m, g, f, cam, env, 0, 0, g->width, g->height);
}

int main(int argc, char *argv[]) {
	glm::mat4 model = glm::mat4(
		0.5, 0.0, 0.0, 0.0,
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0
	);
	Mesh m = Mesh("model.obj", model);

	float theta = 0.1;
	Camera cam = Camera(theta);

	PointLight light;
	light.pos = glm::vec3(0.0, 1.0, 0.0);
	light.falloff = 0.1;
	light.color = glm::vec3(1.0);
	light.radius = 0.2;
	light.shadowMapA = std::vector<float>(WIDTH * HEIGHT, 0.0);
	light.shadowMapB = std::vector<float>(WIDTH * HEIGHT, 0.0);

	float ambient = 0.05;
	std::vector<PointLight> pointLights = std::vector<PointLight>();
	pointLights.push_back(light);
	Environment env = Environment(pointLights, ambient);

	MyWindow w = MyWindow(WIDTH, HEIGHT, false);
	GeometryBuffer g = GeometryBuffer(WIDTH, HEIGHT);

	float alpha = 1.0;

	while (!w.shouldClose) {
		SDL_Event event;
		if (w.window.pollForInputEvents(event)) {
			switch (event.type) {
				case SDL_WINDOWEVENT:
					if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
						w.shouldClose = true;
					}

					// if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					// 	this->width = event.window.data1;
					// 	this->height = event.window.data2;
					// 	std::cout << "WINDOW RESIZED" << std::endl;
					// }

					break;

				case SDL_KEYDOWN:
					if (event.key.keysym.sym == SDLK_LEFT) {
						theta -= 0.01;
					} else if (event.key.keysym.sym == SDLK_RIGHT) {
						theta += 0.01;
					} 

					cam = Camera(theta);

					break;

				case SDL_KEYUP: 
					break;

				case SDL_MOUSEBUTTONDOWN:
					w.window.savePPM("output.ppm");
					w.window.saveBMP("output.bmp");
					break;
			}
		}

		w.clear();
		g.clear();

		traceMeshThreaded(&m, &cam, &g, 8);
		shadowThreaded(&m, &g, &env, alpha, 8);
		lightingThreaded(&m, &g, &w, &cam, &env, 8);

		// traceMesh(&m, &cam, &g);
		// shadow(&m, &g, &env);
		// lighting(&m, &g, &w, &cam, &env);

		alpha = 0.3;

		w.window.renderFrame();

		std::cout << "render" << std::endl;
	}
}
