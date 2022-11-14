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

#define WIDTH 800
#define HEIGHT 800

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
	glm::vec3 uv;
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
		// glm::mat4 view;
		// glm::mat4 projection;
		// glm::vec3 position;
		// float fovy;
		// float aspect;
		// float znear;
		// float zfar;
		float f;
		glm::vec3 pos;
		float pitch;
		float yaw;

	Camera(float theta) {
		this->pos = float(5.0) * glm::vec3(cos(-theta + (PI/2)), 0.0, sin(-theta + (PI/2)));
		this->f = 2.0;
		this->pitch = 0.0;
		this->yaw = theta;//PI / 2;
	}

	// Camera(glm::vec3 pos, glm::vec3 forward, float fovy, float aspect, float znear, float zfar) : position(pos), fovy(fovy), aspect(aspect), znear(znear), zfar(zfar) {
	// 	this->view = glm::lookAt(pos, pos + forward, glm::vec3(0.0, 1.0, 0.0));
	// 	this->projection = glm::perspectiveLH(fovy, aspect, znear, zfar);

	// 	// float sin_fov = sin(0.5 * fovy);
	// 	// float cos_fov = cos(0.5 * fovy);
	// 	// float h = cos_fov / sin_fov;
	// 	// float w = h / aspect;
	// 	// float r = zfar / (zfar - znear);
		
	// 	// glm::mat4 p = glm::mat4(
	// 	// 	glm::vec4(w, 0.0, 0.0, 0.0),
	// 	// 	glm::vec4(0.0, h, 0.0, 0.0),
	// 	// 	glm::vec4(0.0, 0.0, r, 1.0),
	// 	// 	glm::vec4(0.0, 0.0, -r * znear, 0.0)
	// 	// );

	// 	// this->projection = p;
	// }
};

struct Material {
	/// @brief base color of the material rgba
	glm::vec4 color;
	/// @brief name of the material (to look up faces in draw)
	std::string name;
};

struct FaceIndices {
	/// @brief vertex 0 index
	uint32_t v0;
	/// @brief vertex 1 index
	uint32_t v1;
	/// @brief vertex 2 index
	uint32_t v2;
};

struct Face {
	WorldCoord a;
	WorldCoord b;
	WorldCoord c;
	Material mtl;
};

class Mesh {
	public:
		// /// @brief all the materials in the obj file
		// std::vector<Material> materials;
		// /// @brief all the posiitons int the obj file
		// std::vector<WorldCoord> positions;
		// /// @brief all the faces in the obj file (each face is 3 indices into the array of )
		// std::vector<Face> faces;
		// /// @brief map from material name to list of faces to be drawn with that material
		// std::unordered_map<std::string, std::vector<uint32_t>> draw;
		// /// @brief model matrix to scale this model
		// glm::mat4 model;
		std::vector<Face> faces;

	Mesh(std::string path, glm::mat4 model) {
		std::string buf;
		std::ifstream F(path);

		// read mtl file name
		getline(F, buf);	

		std::ifstream MF(split(buf, ' ')[1]);

		std::vector<Material> materials_buf;
		Material m;
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

		std::vector<WorldCoord> positions_buf;
		std::vector<FaceIndices> faces_buf;
		std::string current_mtl;
		bool mtl_set = false;
		std::unordered_map<std::string, std::vector<uint32_t>> draw_buf;

		for (auto m = materials_buf.begin(); m != materials_buf.end(); m++) {
			draw_buf[m->name] = std::vector<uint32_t>();
		}

		while (getline(F, buf)) {
			// std::cout << buf << std::endl;
			if (buf.size() == 0) {
				continue;
			}

			std::vector<std::string> vals = split(buf, ' ');

			if (buf[0] == 'v') {
				float x = std::stof(vals[1]);
				float y = std::stof(vals[2]);
				float z = std::stof(vals[3]);
				glm::vec3 pos = glm::vec3(x, y, z);	
				WorldCoord w;
				w.pos = pos;
				positions_buf.push_back(w);
			} else if (buf[0] == 'f') {
				// obj index from 1, c++ from 0
				uint32_t i0 = std::stoi(vals[1].substr(0, vals[1].size() - 1)) - 1;
				uint32_t i1 = std::stoi(vals[2].substr(0, vals[2].size() - 1)) - 1;
				uint32_t i2 = std::stoi(vals[3].substr(0, vals[3].size() - 1)) - 1;
				FaceIndices f = FaceIndices { i0, i1, i2 };
				faces_buf.push_back(f);
				if (mtl_set) {
					// if material set then append this face to the draw buffer's current materials faces
					draw_buf.at(current_mtl).push_back(faces_buf.size() - 1);
				}
			} else if (vals[0] == "usemtl") {
				// change the current material being used
				mtl_set = true;
				current_mtl = vals[1];
			}
		}

		F.close();

		this->faces = std::vector<Face>(faces.size());
		for (auto mtl = materials_buf.begin(); mtl != materials_buf.end(); mtl++) {
			if (draw_buf.find(mtl->name) != draw_buf.end()) {
				auto faces = draw_buf.at(mtl->name);
				for (auto face_iter = faces.begin(); face_iter != faces.end(); face_iter++) {
					FaceIndices face = faces_buf[*face_iter];

					WorldCoord a = positions_buf[face.v0];
					a.pos = glm::vec3(model * glm::vec4(a.pos.x, a.pos.y, a.pos.z, 1.0));
					WorldCoord b = positions_buf[face.v1];
					b.pos = glm::vec3(model * glm::vec4(b.pos.x, b.pos.y, b.pos.z, 1.0));
					WorldCoord c = positions_buf[face.v2];
					c.pos = glm::vec3(model * glm::vec4(c.pos.x, c.pos.y, c.pos.z, 1.0));

					Face f;
					f.a = a;
					f.b = b;
					f.c = c;
					f.mtl = *mtl;
					this->faces.push_back(f);
				}
			}
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
	glm::vec3 t = world.pos - cam->pos;
	glm::mat3 pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->pitch), -sin(cam->pitch)),
		glm::vec3(0.0, sin(cam->pitch), cos(cam->pitch))
	);
	glm::mat3 yaw_mat = glm::mat3(
		glm::vec3(cos(cam->yaw), 0.0, sin(cam->yaw)),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(cam->yaw), 0.0, cos(cam->yaw))
	);
	DeviceCoord device;
	t = yaw_mat * t;
	t = pitch_mat * t;
	device.pos.x = -cam->f * (t.x / t.z);
	device.pos.y = cam->f * (t.y / t.z);
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
		drawWorldTriangle(face->a, face->b, face->c, face->mtl, f, cam);
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
	glm::vec4 color;
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
		r.color = info.face.mtl.color;
		r.intersection = info;
		return r;
	} else {
		RayResult r;
		r.valid = false;
		return r;
	}
}

inline void traceMeshBounded(Mesh *m, Camera *cam, Framebuffer *f, size_t sx, size_t sy, size_t w, size_t h) {
	glm::mat3 pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->pitch), -sin(cam->pitch)),
		glm::vec3(0.0, sin(cam->pitch), cos(cam->pitch))
	);
	glm::mat3 yaw_mat = glm::mat3(
		glm::vec3(cos(-cam->yaw), 0.0, sin(-cam->yaw)),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(-cam->yaw), 0.0, cos(-cam->yaw))
	);
	glm::mat3 mat = pitch_mat * yaw_mat;
	for (size_t y = sy; y < sy + h; y++) {
		for (size_t x = sx; x < sx + w; x++) {
			Ray ray;
			ray.src = cam->pos;
			ray.dir = glm::normalize(mat * glm::vec3(
					(2 * float(x)) / f->width() - 1,
					-((2 * float(y)) / f->height() - 1),
					-cam->f));
			
			RayResult r = traceRay(ray, m);

			if (r.valid) {
				Ray shadowRay;
				shadowRay.src = ray.src + ray.dir * r.intersection.t;
				glm::vec3 lightPos = glm::vec3(0.0, 1.1, 0.0);
				shadowRay.dir = glm::normalize(lightPos - shadowRay.src);
				shadowRay.src += float(0.01) * shadowRay.dir;
				
				RayResult shadowResult = traceRay(shadowRay, m);

				PixCoord p;
				p.x = x;
				p.y = y;
				p.depth = 0.1;
				
				if (!shadowResult.valid || shadowResult.intersection.t >= glm::length(lightPos - shadowRay.src)) {
					f->putPixel(p, r.color);
				} else {
					f->putPixel(p, r.color * float(0.1));
				}
			}
		}
	}
}

inline void traceMeshTreaded(Mesh *m, Camera *cam, Framebuffer *f, size_t numThreads) {
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(traceMeshBounded, m, cam, f, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

struct PointLight {
	glm::vec3 pos;
	float falloff;
	glm::vec3 color;
};

struct Environment {
	std::vector<PointLight> pointLights;
	float ambient;
};

inline void traceMesh(Mesh *m, Camera *cam, Framebuffer *f, Environment *env) {
	glm::mat3 pitch_mat = glm::mat3(
		glm::vec3(1.0, 0.0, 0.0),
		glm::vec3(0.0, cos(cam->pitch), -sin(cam->pitch)),
		glm::vec3(0.0, sin(cam->pitch), cos(cam->pitch))
	);
	glm::mat3 yaw_mat = glm::mat3(
		glm::vec3(cos(-cam->yaw), 0.0, sin(-cam->yaw)),
		glm::vec3(0.0, 1.0, 0.0),
		glm::vec3(-sin(-cam->yaw), 0.0, cos(-cam->yaw))
	);
	glm::mat3 mat = pitch_mat * yaw_mat;
	for (size_t y = 0; y < f->height(); y++) {
		for (size_t x = 0; x < f->width(); x++) {
			Ray ray;
			ray.src = cam->pos;
			ray.dir = glm::normalize(mat * glm::vec3(
					(2 * float(x)) / f->width() - 1,
					-((2 * float(y)) / f->height() - 1),
					-cam->f));
			
			RayResult r = traceRay(ray, m);
			glm::vec3 albedo = glm::vec3(r.color.x, r.color.y, r.color.z);

			glm::vec3 result = glm::vec3(0.0);
			result += albedo * env->ambient;

			for (auto &l : env->pointLights) {
				glm::vec3 toLight = l.pos - (ray.src + ray.dir * r.intersection.t);
				float d2 = toLight.x * toLight.x + toLight.y * toLight.y + toLight.z * toLight.z;
				float attenuation = 1.0 / (d2 * l.falloff);

				Face face = r.intersection.face;
				glm::vec3 e1 = face.b.pos - face.a.pos;
				glm::vec3 e2 = face.c.pos - face.a.pos;

				glm::vec3 normal = glm::cross(e1, e2);

				if (r.valid) {
					Ray shadowRay;
					shadowRay.src = ray.src + ray.dir * r.intersection.t;
					shadowRay.dir = glm::normalize(l.pos - shadowRay.src);
					shadowRay.src += float(0.001) * shadowRay.dir;
					
					RayResult shadowResult = traceRay(shadowRay, m);
					
					if (!shadowResult.valid || shadowResult.intersection.t >= glm::length(l.pos - shadowRay.src)) {
						result += albedo * attenuation;
					}
				}
			}

			PixCoord p;
			p.x = x;
			p.y = y;
			p.depth = 0.1;

			f->putPixel(p, glm::vec4(result.x, result.y, result.z, 1.0));
		}
	}
}

int main(int argc, char *argv[]) {
	glm::mat4 model = glm::mat4(
		0.5, 0.0, 0.0, 0.0, 
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0
	);
	Mesh m = Mesh("cornell-box.obj", model);

	float theta = 0.0;
	Camera cam = Camera(theta);

	PointLight light;
	light.pos = glm::vec3(0.0, 1.0, 0.0);
	light.falloff = 1.0;
	light.color = glm::vec3(1.0);

	Environment env;
	env.ambient = 0.1;
	env.pointLights = std::vector<PointLight>();
	env.pointLights.push_back(light);

	MyWindow w = MyWindow(WIDTH, HEIGHT, false);

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

		w.window.clearPixels();
		w.depth = std::vector<float>(w.width() * w.height(), -1000.0);

		// drawMesh(&m, &cam, &w);
		traceMesh(&m, &cam, &w, &env);
		// traceMeshTreaded(&m, &cam, &w, 8);

		w.window.renderFrame();
	}
}
