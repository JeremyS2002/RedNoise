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
#include <list>
#include <utility>
#include <limits.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#define WIDTH 640
#define HEIGHT 480
// #define WIDTH 800
// #define HEIGHT 800
#define MAX_BOUNCES 10

#define PI 3.14159265

static constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5;

// https://pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error#gamma
inline constexpr float gamma(int n) {
	return (n * MachineEpsilon) / (1.0 - n * MachineEpsilon);
}

template <typename T>
inline T interpUV(T a, T b, T c, float u, float v) {
	T d = a + (b - a) + (c - a);
	return a*(float(1.0)-u)*(float(1.0)-v) + b*u*(float(1.0)-v) + c*(float(1.0)-u)*v + d*u*v;
}

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
		Camera() {
			float theta = 0.0;
			this->pos = float(5.0) * glm::vec3(cos(-theta + (PI/2)), 0.0, sin(-theta + (PI/2)));
			this->f = 2.0;
			this->pitch = 0.0;
			this->yaw = theta;
		}

		Camera(float theta) {
			this->pos = float(5.0) * glm::vec3(cos(-theta + (PI/2)), 0.0, sin(-theta + (PI/2)));
			this->f = 2.0;
			this->pitch = 0.0;
			this->yaw = theta;
		}

		Camera(glm::vec3 p, glm::vec3 dir) {
			this->pos = p;
			this->f = 2.0;
			dir = glm::normalize(dir);
			this->pitch = asinf32(dir.y);
			this->yaw = acosf32(dir.z / cosf32(pitch));
		}

		Camera(glm::vec3 p) {
			this->pos = p;
			this->f = 2.0;
			glm::vec3 d = glm::normalize(p);
			this->pitch = asinf32(d.y);
			this->yaw = acosf32(d.z / cosf32(pitch));
		}

		Camera(glm::vec3 position, float p, float y) {
			this->pos = position;
			this->f = 2.0;
			this->pitch = p;
			this->yaw = y;
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

		void setPos(glm::vec3 newPos) {
			this->pos = newPos;
		}

		void setPitch(float newPitch) {
			this->pitch = newPitch;
		}

		void setYaw(float newYaw) {
			this->yaw = newYaw;
		}

		void addPos(glm::vec3 deltaPos) {
			this->pos += deltaPos;
		}

		void addPitch(float deltaPitch) {
			this->pitch += deltaPitch;
		}

		void addYaw(float deltaYaw) {
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
};

struct Material {
	/// @brief base color of the material rgba
	glm::vec4 albedo;
	int albedo_map;
	/// @brief how much specular components of lighting contribute to the render
	float roughness;
	/// @brief how metallic the material is
	float metallic;
	/// @brief how much each channel of light will be refracted
	float transmission;
	/// @brief index of refraction
	float ior;
	/// @brief how much each channel is reflected
	float reflectance;
	/// @brief name of the material (to look up faces in draw)
	int normal_map;
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

struct Ray {
	glm::vec3 src;
	glm::vec3 dir;
};

struct Intersection {
	Face face;
	float t;
	float u;
	float v;

	glm::vec3 getNormal() {
		if (face.hasNormals) {
			return glm::normalize(interpUV(face.a.normal, face.b.normal, face.c.normal, u, v));
		} else {
			glm::vec3 e1 = face.b.pos - face.a.pos;
			glm::vec3 e2 = face.c.pos - face.a.pos;
			return glm::normalize(glm::cross(e1, e2));
		}
	}

	glm::vec3 getTangent() {
		if (!face.hasUvs) {
			return face.b.pos - face.a.pos;
		}
		glm::vec3 e1 = face.b.pos - face.a.pos;
		glm::vec3 e2 = face.c.pos - face.a.pos;

		glm::vec2 duv1 = face.b.uv - face.a.uv;
		glm::vec2 duv2 = face.c.uv - face.a.uv;

		float f = 1.0 / (duv1.x * duv2.y - duv2.x * duv1.y);

		glm::vec3 t;
		t.x = f * (duv2.y * e1.x - duv1.y * e2.x);
		t.y = f * (duv2.y * e1.y - duv1.y * e2.y);
		t.z = f * (duv2.y * e1.z - duv1.y * e2.z);

		return t;
	}

	glm::vec2 getUV() {
		if (face.hasUvs) {
			return interpUV(face.a.uv, face.b.uv, face.c.uv, u, v);
		} else {
			return glm::vec2(0.0);
		}
	}

	int getMaterial() {
		return face.mtl;
	}
};

inline bool rayTriangleIntersection(Ray ray, Face face, Intersection *isect) {
	// std::cout << "ray src: " << ray.src.x << " " << ray.src.y << " " << ray.src.z << std::endl;
	// std::cout << "ray dir: " << ray.dir.x << " " << ray.dir.y << " " << ray.dir.z << std::endl;

	// std::cout << "a pos  : " << a.pos.x << " " << a.pos.y << " " << a.pos.z << std::endl;
	// std::cout << "b pos  : " << b.pos.x << " " << b.pos.y << " " << b.pos.z << std::endl;
	// std::cout << "c pos  : " << c.pos.x << " " << c.pos.y << " " << c.pos.z << std::endl;
	auto a = face.a;
	auto b = face.b;
	auto c = face.c;

	glm::vec3 e0 = b.pos - a.pos;
	glm::vec3 e1 = c.pos - a.pos;
	glm::vec3 SPVector = ray.src - a.pos;
	glm::mat3 DEMatrix(-ray.dir, e0, e1);
	if (glm::determinant(DEMatrix) == 0.0) {
		// ray parallel to plane
		return false;
	}
	glm::vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;
	isect->face = face;
	isect->t = possibleSolution.x;
	isect->u = possibleSolution.y;
	isect->v = possibleSolution.z;

	if (isect->u < 0.0 || isect->u > 1.0) {
		return false;
	}

	if (isect->v < 0.0 || isect->v > 1.0) {
		return false;
	}

	if (isect->u + isect->v > 1.0) {
		return false;
	}

	if (isect->t <= 0.0) {
		return false;
	}

	return true;
}

class Intersectable {
	public: 
		virtual bool Intersect(const Ray &ray, Intersection *isect) = 0;
		virtual const Material *material(int idx) = 0;
		virtual const TextureMap *texture(int idx) = 0;
};

class Mesh: public Intersectable {
	public:
		bool Intersect(const Ray &ray, Intersection *isect) {
			// std::vector<Intersection> intersections;// = std::vector<Intersection>(m->faces.size());
			bool intersectionFound = false;
			Intersection info;

			for (auto face = faces.begin(); face != faces.end(); face++) {
				Intersection i;
				if (rayTriangleIntersection(ray, *face, &i)) {
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
				(*isect) = info;
				return true;
			} else {
				return false;
			}
		}

		const Material *material(int idx) {
			return &materials[idx];
		}

		const TextureMap *texture(int idx) {
			return &textures[idx];
		}

		std::vector<Face> faces;
		std::vector<Material> materials;
		std::vector<TextureMap> textures;

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
		std::vector<TextureMap> textures_buf;
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
				m.transmission = 0.0;
				m.reflectance = 0.0;
				m.ior = 1.45;
				m.roughness = 0.5;
				m.metallic = 0.0;
				m.albedo_map = -1;
				m.normal_map = -1;
			} else if (vals[0] == "Kd") {
				if (vals.size() < 4) {
					m.albedo = glm::vec4(0.5);
					m.albedo_map = textures_buf.size();
					textures_buf.push_back(TextureMap(vals[1]));
				} else {
					float r = std::stof(vals[1]);
					float g = std::stof(vals[2]);
					float b = std::stof(vals[3]);
					float a = 1.0;
					if (vals.size() >= 5) {
						a = std::stof(vals[4]);
					}
					m.albedo = glm::vec4(r, g, b, a);
				}
			} else if (vals[0] == "R") {
				m.roughness = std::stof(vals[1]);
			} else if (vals[0] == "T") {
				float t = std::stof(vals[1]);
				m.transmission = t;
			} else if (vals[0] == "Rf") {
				float r = std::stof(vals[1]);
				m.reflectance = r;
			} else if (vals[0] == "IOR") {
				float ior = std::stof(vals[1]);
				m.ior = ior;
			} else if (vals[0] == "M") {
				float metallic = std::stof(vals[1]);
				m.metallic = metallic;
			} else if (vals[0] == "Norm") {
				m.normal_map = textures_buf.size();
				textures_buf.push_back(TextureMap(vals[1]));
			}
		}

		materials_buf.push_back(m);

		MF.close();

		this->materials = materials_buf;
		this->textures = textures_buf;

		std::vector<glm::vec3> positions_buf;
		std::vector<glm::vec3> normals_buf;	
		std::vector<glm::vec2> uvs_buf;
		std::vector<FaceIndices> faces_buf;
		std::string current_mtl;
		bool mtl_set = false;
		std::unordered_map<std::string, int> material_name_map;

		for (int i = 0; i < materials.size(); i++) {
			material_name_map[materials[i].name] = i;
			std::cout << "name: " << materials[i].name << std::endl;
			std::cout << "roughness: " << materials[i].roughness << std::endl;
			std::cout << "transmission: " << materials[i].transmission << std::endl;
			std::cout << "reflectance: " << materials[i].reflectance << std::endl;
			std::cout << "metallic: " << materials[i].metallic << std::endl;
			std::cout << "ior: " << materials[i].ior << std::endl;
			std::cout << "albedo: " << materials[i].albedo.r << " " << materials[i].albedo.g << " " << materials[i].albedo.b << std::endl;
			std::cout << std::endl;
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
	device.pos.x = -cam->getF() * ((t.x * (float(HEIGHT) / float(WIDTH))) / t.z);
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
		f->putPixelNoDepth(draw, mtl.albedo);
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
inline void drawWorldTriangleWireframe(WorldCoord a, WorldCoord b, WorldCoord c, Material mtl, Framebuffer *f, Camera *cam) {
	auto device_a = world2device(a, cam);
	auto device_b = world2device(b, cam);
	auto device_c = world2device(c, cam);
	
	drawDeviceTriangleWireFrame(device_a, device_b, device_c, mtl, f);
}

inline void __internalflatPixTriangle(PixCoord a, PixCoord b, PixCoord c, Material mtl, Framebuffer *f) {
	// std::cout << a.x << " " << a.y << std::endl;
	// std::cout << b.x << " " << b.y << std::endl;
	// std::cout << c.x << " " << c.y << std::endl;
	// std::cout << std::endl;
	if (a.y == b.y && a.y == c.y) {
		return;
	}

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
			
			f->putPixel(p, mtl.albedo);

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

void drawMeshWireframe(Mesh *m, Camera *cam, Framebuffer *f) {
	for (auto &face : m->faces) {
		drawWorldTriangleWireframe(face.a, face.b, face.c, m->materials[face.mtl], f, cam);
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

// https://pbr-book.org/3ed-2018/Utilities/Memory_Management#MemoryArena
class MemoryArena {
	public:
		MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) { }
		~MemoryArena() {
			free(currentBlock);
			for (auto &block : usedBlocks) {
				free(block.second);
			}
			for (auto &block : availableBlocks) {
				free(block.second);
			}
		}

		void *Alloc(size_t nBytes) {
			// round to machine size (don't know if necissary since not using allocAllgned but can't hurt)
			nBytes = ((nBytes + 15) & (~15));
			// if not enough space in current block
			if (currentBlockPos + nBytes > currentAllocSize) {
				// and current block is allocated
				if (currentBlock) {
					usedBlocks.push_back(std::make_pair(currentAllocSize, currentBlock));
					currentBlock = nullptr;
				}

				// try to find 
				for (auto iter = availableBlocks.begin(); iter != availableBlocks.end(); iter++) {
					if (iter->first >= nBytes) {
						currentAllocSize = iter->first;
						currentBlock = iter->second;
						availableBlocks.erase(iter);
						break;
					}
				}

				// if didn't find one
				if (!currentBlock) {
					currentAllocSize = std::max(nBytes, blockSize);
					currentBlock = (uint8_t*)malloc(currentAllocSize);
				}
				currentBlockPos = 0;
			}
			void *ret = currentBlock + currentBlockPos;
			currentBlockPos += nBytes;
			return ret;
		}

		template<typename T>
		T *Alloc(size_t n = 1, bool runConstructor = true) {
			T *ret = (T *)Alloc(n * sizeof(T));
			if (runConstructor) {
				for (size_t i = 0; i < n; i++) {
					new (&ret[i]) T();
				}
			}
			return ret;
		}

		void Reset() {
			currentBlockPos = 0;
			availableBlocks.splice(availableBlocks.begin(), usedBlocks);
		}

		size_t TotalAllocated() const {
			size_t total = currentAllocSize;
			for (const auto &alloc : usedBlocks) {
				total += alloc.first;
			}
			for (const auto &alloc : availableBlocks) {
				total += alloc.first;
			}
			return total;
		}

	private:
		const size_t blockSize;
		size_t currentBlockPos = 0;
		size_t currentAllocSize = 0;
		uint8_t *currentBlock = nullptr;
		std::list<std::pair<size_t, uint8_t *>> usedBlocks;
		std::list<std::pair<size_t, uint8_t *>> availableBlocks;
};

// https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#Bounds3::MaximumExtent
struct AABB {
	glm::vec3 Offset(const glm::vec3 &p) const {
		glm::vec3 o = p - min;
		if (max.x > min.x) o.x /= max.x - min.x;
		if (max.y > min.y) o.y /= max.y - min.y;
		if (max.z > min.z) o.z /= max.z - min.z;
		return o;
	}

	glm::vec3 Diagonal() const {
		return max - min;
	}

	int MaximumExtent() const {
		glm::vec3 d = Diagonal();
		if (d.x > d.y && d.x > d.z) {
			return 0;
		} else if (d.y > d.z) {
			return 1;
		} else {
			return 2;
		}
	}

	float SurfaceArea() const {
		glm::vec3 d = Diagonal();
		return 2.0* (d.x * d.y + d.x * d.z + d.z * d.z);
	}

	float Volume() const {
		glm::vec3 d = Diagonal();
		return d.x * d.y * d.z;
	}

	inline bool Intersect(const Ray &ray, float *hitt0, float *hitt1) const {
		float t0 = 0;
		float t1 = std::numeric_limits<float>::max();
		// for each axis
		for (int i = 0; i < 3; i++) {
			float invRayDir = 1.0 / ray.dir[i];
			float tNear = (min[i] - ray.src[i]) * invRayDir;
			float tFar = (max[i] - ray.src[i]) * invRayDir;
			if (tNear > tFar) {
				std::swap(tNear, tFar);
			}
			// TODO test without this
			tFar *= 1 + 2 * gamma(3);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar < t1 ? tFar : t1;
			if (t0 > t1) return false;
		}

		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	inline bool Intersect(const Ray &ray, const glm::vec3 &invDir, int dirIsNeg[3]) const {
		const AABB &aabb = *this;
		float tMin = (aabb[dirIsNeg[0]].x - ray.src.x) * invDir.x;
		float tMax = (aabb[1-dirIsNeg[0]].x - ray.src.x) * invDir.x;
		float tyMin = (aabb[dirIsNeg[1]].y - ray.src.y) * invDir.y;
		float tyMax = (aabb[1-dirIsNeg[1]].y - ray.src.y) * invDir.y;

		tMax *= 1 + 2 * gamma(3);
		tMin *= 1 + 2 * gamma(3);

		if (tMin > tyMax || tyMin > tMax) {
			return false;
		}
		if (tyMin > tMin) tMin = tyMin;
		if (tyMax < tMax) tMax = tyMax;

		float tzMin = (aabb[dirIsNeg[2]].z - ray.src.z) * invDir.z;
		float tzMax = (aabb[1-dirIsNeg[2]].z - ray.src.z) * invDir.z;
		tzMax *= 1 + 2 * gamma(3);
		if (tMin > tzMax || tzMin > tMax) {
			return false;
		}
		if (tzMin > tMin) tMin = tzMin;
		if (tzMax < tMax) tMax = tzMax;
		return tMax > 0;
	}

	const glm::vec3 &operator[](int i) const {
		if (i == 0) {
			return min;
		} else {
			return max;
		}
	};
	glm::vec3 &operator[](int i) {
		if (i == 0) {
			return min;
		} else {
			return max;
		}
	};

	glm::vec3 min;
	glm::vec3 max;
};

AABB Union(AABB l, AABB r) {
	AABB n;
	n.min = glm::min(l.min, r.min);
	n.max = glm::max(l.max, r.max);
	return n;
}

AABB Union(AABB b, glm::vec3 p) {
	AABB n;
	n.min = glm::min(b.min, p);
	n.max = glm::max(b.max, p);
	return n;
}

inline AABB bound(Face *f) {
	AABB n;
	n.min = glm::min(f->a.pos, glm::min(f->b.pos, f->c.pos));
	n.max = glm::max(f->a.pos, glm::max(f->b.pos, f->c.pos));
	return n;
}

struct BVHBuildNode {
	void InitLeaf(int first, int n, const AABB &b) {
		firstFaceOffset = first;
		nFaces = n;
		aabb = b;
		children[0] = nullptr;
		children[1] = nullptr;
	}

	void InitInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
		children[0] = c0;
		children[1] = c1;
		aabb = Union(c0->aabb, c1->aabb);
		splitAxis = axis;
		nFaces = 0;
	}

	AABB aabb;
	BVHBuildNode *children[2];
	int splitAxis;
	int firstFaceOffset;
	int nFaces;
};

struct BVHFaceInfo {
	size_t i;
	AABB aabb;
	glm::vec3 centroid;
};

struct BVHLinearNode {
	AABB aabb;
	// why union when both are ints?
	union {
		// leaf
		int facesOffset;
		// interior
		int secondChildOffset;
	};
	// 0 => interior
	uint16_t nFaces;
	// only for interior nodes (xyz) = (012)
	uint8_t axis;
	// padding to 32 byte size
	// if first node is cache aligned then all subsequent will be
	// no node will straddle cache lines (very smart)
	// probably useless though since i'm not allocating things alligned
	uint8_t pad[1];
};

// https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
class BVH: public Intersectable {
	public:
		BVH(Mesh *m, int maxFacesInNode) : maxFacesInNode(maxFacesInNode), faces(m->faces), materials(m->materials), textures(m->textures){
			if (faces.size() == 0) {
				return;
			}

			std::vector<BVHFaceInfo> faceInfo(faces.size());
			for (size_t i = 0; i < faces.size(); i++) {
				BVHFaceInfo info;
				info.i = i;
				info.aabb = bound(&faces[i]);
				info.centroid = (info.aabb.min + info.aabb.max) / float(2.0);
				faceInfo[i] = info;
			}

			MemoryArena arena(1024 * 1024);
			int totalNodes = 0;
			std::vector<Face> orderedFaces;

			BVHBuildNode *root = recursiveBuild(arena, faceInfo, 0, faces.size(), &totalNodes, orderedFaces);

			faces.swap(orderedFaces);

			nodes = (BVHLinearNode*)malloc(totalNodes * sizeof(BVHLinearNode));
			int offset = 0;
			flattenTree(root, &offset);
		}

		~BVH() {
			free(this->nodes);
		}

		bool Intersect(const Ray &ray, Intersection *isect) {
			bool hit = false;
			float closest_hit = std::numeric_limits<float>::max();
			glm::vec3 invDir = glm::vec3(1.0 / ray.dir.x, 1.0 / ray.dir.y, 1.0 / ray.dir.z);
			int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

			int toVisitOffset = 0;
			int currentNodeIndex = 0;
			int nodesToVisit[64];
			while (true) {
				const BVHLinearNode *node = &nodes[currentNodeIndex];
				if (node->aabb.Intersect(ray, invDir, dirIsNeg)) {
					if (node->nFaces > 0) {
						// then leaf node
						for (int i = 0; i < node->nFaces; i++) {
							Intersection test_isect;
							if (rayTriangleIntersection(ray, faces[node->facesOffset+i], &test_isect)) {
								hit = true;
								if (test_isect.t < closest_hit) {
									closest_hit = test_isect.t;
									(*isect) = test_isect;
								}
							}
						}
						if (toVisitOffset == 0) break;
						currentNodeIndex = nodesToVisit[--toVisitOffset];
					} else {
						if (dirIsNeg[node->axis]) {
							nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
							currentNodeIndex = node->secondChildOffset;
						} else {
							nodesToVisit[toVisitOffset++] = node->secondChildOffset;
							currentNodeIndex = currentNodeIndex + 1;
						}
					}
				} else {
					if (toVisitOffset == 0) break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				}
			}
			return hit;
		}

		const Material *material(int idx) {
			return &materials[idx];
		}

		const TextureMap *texture(int idx) {
			return &textures[idx];
		}

	private:
		BVHBuildNode *recursiveBuild(
			MemoryArena &arena, 
			std::vector<BVHFaceInfo> &faceInfo, 
			int start, 
			int end,
			int *totalNodes,
			std::vector<Face> &orderedFaces
		) {
			BVHBuildNode *node = arena.Alloc<BVHBuildNode>();
			(*totalNodes)++;
			AABB aabb;
			for (int i = start; i < end; i++) {
				aabb = Union(aabb, faceInfo[i].aabb);
			}

			int nFaces = end - start;
			if (nFaces == 1) {
				// if only one face must be a leaf
				int firstFaceOffset = orderedFaces.size();
				for (int i = start; i < end; i++) {
					int faceNum = faceInfo[i].i;
					orderedFaces.push_back(faces[faceNum]);
				}
				node->InitLeaf(firstFaceOffset, nFaces, aabb);
				return node;
			} else {
				AABB centroidAABB;
				centroidAABB.min = faceInfo[start].centroid;
				centroidAABB.max = faceInfo[start].centroid;
				for (int i = start+1; i < end; i++) {
					centroidAABB = Union(centroidAABB, faceInfo[i].centroid);
				}
				int dim = centroidAABB.MaximumExtent();

				int mid = (start + end) / 2;
				
				if (centroidAABB.max[dim] == centroidAABB.min[dim]) {
					// if the bounds have 0 volume then make a leaf
					// kinda strange scenario
					int firstFaceOffset = orderedFaces.size();
					for (int i = start; i < end; i++) {
						int faceNum = faceInfo[i].i;
						orderedFaces.push_back(faces[faceNum]);
					}
					node->InitLeaf(firstFaceOffset, nFaces, aabb);
					return node;
				} else {
					// split by surface area heuristic
					if (nFaces <= 4) {
						mid = (start + end) / 2;
						std::nth_element(&faceInfo[start], &faceInfo[mid], &faceInfo[end-1]+1,
							[dim](const BVHFaceInfo &a, const BVHFaceInfo &b) {
								return a.centroid[dim] < b.centroid[dim];
							});
					} else {
						constexpr int nBuckets = 12;
						struct BucketInfo {
							int count = 0;
							AABB aabb;
						};
						BucketInfo buckets[nBuckets];

						for (int i = start; i < end; i++) {
							int b = nBuckets * centroidAABB.Offset(faceInfo[i].centroid)[dim];
							if (b == nBuckets) b = nBuckets - 1;
							buckets[b].count++;
							buckets[b].aabb = Union(buckets[b].aabb, faceInfo[i].aabb);
						}

						float costs[nBuckets - 1];
						for (int i = 0; i < nBuckets - 1; i++) {
							AABB b0, b1;
							int count0 = 0, count1 = 0;
							for (int j = 0; j <= i; j++) {
								b0 = Union(b0, buckets[j].aabb);
								count0 += buckets[j].count;
							}
							for (int j = i+1; j < nBuckets; j++) {
								b1 = Union(b1, buckets[j].aabb);
								count1 += buckets[j].count;
							}
							costs[i] = 0.125 + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / aabb.SurfaceArea();
						}

						float minCost = costs[0];
						int minCostSplitBucket = 0;
						for (int i = 1; i < nBuckets - 1; i++) {
							if (costs[i] < minCost) {
								minCost = costs[i];
								minCostSplitBucket = i;
							}
						}

						float leafCost = nFaces;
						if (nFaces > maxFacesInNode || minCost < leafCost) {
							// if have to split or better to split then make an interior node
							BVHFaceInfo *fmid = std::partition(&faceInfo[start], &faceInfo[end-1]+1,
								[=](const BVHFaceInfo &fi) {
									int b = nBuckets * centroidAABB.Offset(fi.centroid)[dim];
									if (b == nBuckets) b = nBuckets - 1;
									return b <= minCostSplitBucket;
								});
							mid = fmid - &faceInfo[0];
						} else {
							// otherwise make a leaf
							int firstFaceOffset = orderedFaces.size();
							for (int i = start; i < end; i++) {
								int faceNum = faceInfo[i].i;
								orderedFaces.push_back(faces[faceNum]);
							}
							node->InitLeaf(firstFaceOffset, nFaces, aabb);
							return node;
						}
					}
					node->InitInterior(
						dim, 
						recursiveBuild(arena, faceInfo, start, mid, totalNodes, orderedFaces),
						recursiveBuild(arena, faceInfo, mid, end, totalNodes, orderedFaces)
					);
				}
			}
			return node;
		}

		int flattenTree(BVHBuildNode *node, int *offset) {
			BVHLinearNode *linearNode = &nodes[*offset];
			linearNode->aabb = node->aabb;
			int myOffset = (*offset)++;
			if (node->nFaces > 0) {
				linearNode->facesOffset = node->firstFaceOffset;
				linearNode->nFaces = node->nFaces;
			} else {
				linearNode->axis = node->splitAxis;
				linearNode->nFaces = 0;
				flattenTree(node->children[0], offset);
				linearNode->secondChildOffset = flattenTree(node->children[1], offset);
			}
			return myOffset;
		}

		const int maxFacesInNode;
		std::vector<Face> faces;
		BVHLinearNode *nodes = nullptr;
		std::vector<Material> materials;
		std::vector<TextureMap> textures;
};

struct PointLight {
	glm::vec3 pos;
	float radius;
	float falloff;
	glm::vec3 color;
	std::vector<float> shadowMap;
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

inline glm::vec3 physically_based_point_light_calc(
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

	if (glm::dot(normal, view) < 0.0) {
		return glm::vec3(0.0);
	}

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

// https://docs.gl/sl4/reflect
inline glm::vec3 reflect(glm::vec3 I, glm::vec3 N) {
	return I - float(2.0) * glm::dot(N, I) * N;
}

inline glm::vec3 phong_point_light_calc(
	PointLight light,
	glm::vec3 view_pos,
	glm::vec3 world_pos,
	glm::vec3 normal,
	glm::vec3 albedo,
	float roughness,
	float metallic
) {
	glm::vec3 light_dir = glm::normalize(light.pos - world_pos);
	glm::vec3 view_dir = glm::normalize(view_pos - world_pos);

	if (glm::dot(normal, view_dir) < 0.0) {
		return glm::vec3(0.0);
	}

	float diff = std::fmax(glm::dot(normal, light_dir), 0.0);

	glm::vec3 reflect_dir = glm::normalize(reflect(-light_dir, normal));
	float spec = powf32(std::fmax(glm::dot(view_dir, reflect_dir), 0.0), 1.0 - roughness);

	glm::vec3 to_light = light.pos - world_pos;
	float distance2 = glm::dot(to_light, to_light);
	float attenuation = 1.0 / (0.0001 + light.falloff * distance2);

	glm::vec3 diffuse = attenuation * float(0.25) * light.color * diff * albedo;
	glm::vec3 specular = attenuation * float(0.125) * light.color * spec * albedo;

	return diffuse + specular;
}

inline glm::vec3 basic_point_light_calc(
	PointLight light,
	glm::vec3 view_pos,
	glm::vec3 world_pos,
	glm::vec3 normal,
	glm::vec3 albedo,
	float roughness,
	float metallic
) {
	glm::vec3 light_dir = glm::normalize(light.pos - world_pos);
	glm::vec3 view_dir = glm::normalize(view_pos - world_pos);

	if (glm::dot(normal, view_dir) < 0.0) {
		return glm::vec3(0.0);
	}

	float diff = std::fmax(glm::dot(normal, light_dir), 0.0);

	glm::vec3 to_light = light.pos - world_pos;
	float distance2 = glm::dot(to_light, to_light);
	float attenuation = 1.0 / (0.0001 + light.falloff * distance2);

	glm::vec3 diffuse = attenuation * float(0.25) * light.color * diff * albedo;

	return diffuse;
}

class GeometryBuffer {
	public:
		size_t width;
		size_t height;
		std::vector<glm::vec3> world_pos;
		std::vector<glm::vec3> normal;
		std::vector<glm::vec3> tangent;
		std::vector<glm::vec2> uv;
		std::vector<int> material;
		std::vector<float> depth;

		GeometryBuffer(size_t width, size_t height) {
			this->width = width;
			this->height = height;
			this->world_pos = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->normal = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));			this->normal = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->tangent = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->uv = std::vector<glm::vec2>(width * height, glm::vec2(0.0));
			this->depth = std::vector<float>(this->width * this->height, 100000000.0);
			this->material = std::vector<int>(this->width * this->height, -1);
		}

		void clear() {
			this->world_pos = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->normal = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->tangent = std::vector<glm::vec3>(this->width * this->height, glm::vec3(0.0));
			this->uv = std::vector<glm::vec2>(width * height, glm::vec2(0.0));
			this->depth = std::vector<float>(this->width * this->height, 100000000.0);
			this->material = std::vector<int>(this->width * this->height, -1);			
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

inline void traceBounded(Intersectable *m, Camera *cam, GeometryBuffer *g, size_t sx, size_t sy, size_t w, size_t h) {
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

	for (size_t y = sy; y < sy + h; y++) {
		for (size_t x = sx; x < sx + w; x++) {
			Ray ray;
			ray.src = cam->getPos();
			ray.dir = glm::normalize(mat * glm::vec3(
					((2 * float(x)) / g->width - 1) * (float(g->width) / float(g->height)),
					-((2 * float(y)) / g->height - 1),
					-cam->getF()));
			
			Intersection isect;
			if (m->Intersect(ray, &isect)) {
				glm::vec3 world_pos = ray.src + ray.dir * isect.t;

				glm::vec3 normal = isect.getNormal();
				glm::vec2 uv = isect.getUV();
				int mtl_idx = isect.getMaterial();

				size_t index = g->index(x, y);
				g->world_pos[index] = world_pos;
				g->normal[index] = normal;
				g->uv[index] = uv;
				g->material[index] = mtl_idx;
				g->depth[index] = isect.t;
				g->tangent[index] = isect.getTangent();
			}
		}
	}
}

inline void traceThreaded(Intersectable *m, Camera *cam, GeometryBuffer *g, size_t numThreads) {
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(traceBounded, m, cam, g, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

inline void trace(Intersectable *m, Camera *cam, GeometryBuffer *g) {
	traceBounded(m, cam, g, 0, 0, g->width, g->height);
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

std::vector<float> denoise(std::vector<float> &src, GeometryBuffer *g) {
	std::vector<float> result = std::vector<float>(WIDTH * HEIGHT, 0.0);

	for (size_t y = 0; y < HEIGHT; y++) {
		for (size_t x = 0; x < WIDTH; x++) {
			size_t index = g->index(x, y);

			// std::cout << "denoise" << std::endl;
			float dzdx;
			if (x > 0 && x < g->width - 1) {
				dzdx = ((g->depth[index] - g->depth[g->index(x-1, y)]) + (g->depth[g->index(x+1, y)] - g->depth[index])) / 2.0;
			} else if (x > 0) {
				dzdx = g->depth[index] - g->depth[g->index(x-1, y)];
			} else {
				dzdx = g->depth[g->index(x+1, y)] - g->depth[index];
			}

			float dzdy;
			if (y > 0 && y < g->height - 1) {
				dzdy = ((g->depth[index] - g->depth[g->index(x, y-1)]) + (g->depth[g->index(x, y+1)] - g->depth[index])) / 2.0;
			} else if (y > 0) {
				dzdy = g->depth[index] - g->depth[g->index(x, y-1)];
			} else {
				dzdy = g->depth[g->index(x, y+1)] - g->depth[index];
			}
		
			constexpr int ksize = 3;
			// int vals_size = 0;
			// int vals[ksize*ksize];

			float newVal = 0.0;
			float total = 0.0;

			for (int i = -ksize; i <= ksize; i++) {
				for (int j = -ksize; j <= ksize; j++) {
					// std::cout << "BEFORE" << std::endl;
					if (x + i < 0 || x + i >= g->width || y + j < 0 || y + j >= g->height) {
						continue;
					}
					// std::cout << "HERE" << std::endl;
					size_t tmp_idx = g->index(x+i, y+j);
					float dz = std::abs(g->depth[tmp_idx] - g->depth[index]);
					if (dz < float(i) * dzdx + float(j) * dzdy + 0.001) {
						// insert val into vals
						// find index to insert new val
						// int k = 0;
						// for (k = 0; k < vals_size; ++k) {
						// 	if (vals[k] < src[tmp_idx]) {
						// 		break;
						// 	}
						// }
						// for (int l = vals_size; l > k; l++) {
						// 	vals[l] = vals[l-1];
						// }
						// vals[k] = src[tmp_idx];
						// vals_size++;
						// std::cout << vals_size << std::endl;
						newVal += src[tmp_idx];
						total += 1.0;
					}
				}
			}

			// int mid = vals_size / 2;
			// float newVal = vals[mid];

			// std::cout << newVal / total << std::endl;
			result[index] = newVal / total;
			// std::cout << result[index] << std::endl;
		}
	}
	return result;
}

// https://medium.com/@alexander.wester/ray-tracing-soft-shadows-in-real-time-a53b836d123b
inline void shadowBounded(Intersectable *m, GeometryBuffer *g, Environment *env, size_t sx, size_t sy, size_t w, size_t h) {
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
				int samples = 20;
				for (int i = 0; i < samples; i++) {
					Ray ray;
					ray.dir = getConeSample(toLight, angle);
					ray.src = world_pos + float(0.01) * normal;

					Intersection isect;
					if (m->Intersect(ray, &isect)) {
						if (isect.t < glm::length(light.pos - world_pos)) {
							density += 1.0 - m->material(isect.face.mtl)->transmission;
						}
					}
				}

				light.shadowMap[index] = (density / float(samples));
			}
		}
	}
}

void shadowThreaded(Intersectable *m, GeometryBuffer *g, Environment *env, size_t numThreads) {
	for (auto &light : env->pointLights) {
		light.shadowMap = std::vector<float>(WIDTH * HEIGHT, 0.0);
	}
	
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(shadowBounded, m, g, env, x, y, w, h);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}

	for (auto &light : env->pointLights) {
		auto tmp = denoise(light.shadowMap, g);
		light.shadowMap.swap(tmp);
	}
}

void shadow(Intersectable *m, GeometryBuffer *g, Environment *env) {
	for (auto &light : env->pointLights) {
		light.shadowMap = std::vector<float>(WIDTH * HEIGHT, 0.0);
	}
	shadowBounded(m, g, env, 0, 0, g->width, g->height);
	for (auto &light : env->pointLights) {
		auto tmp = denoise(light.shadowMap, g);
		light.shadowMap.swap(tmp);
	}
}

glm::vec3 tonemapPart(glm::vec3 x) {
	// tonemapping https://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
	float t = 1.0 / 2.2;
	// shoulder
	float a = powf32(0.22, t);
	// linear strength
	float b = powf32(0.3, t);
	// linear angle
	float c = powf32(0.1, t);
	// toe strength
	float d = powf32(0.2, t);
	// toe numerator
	float e = powf32(0.01, t);
	// toe denominator
	float f = powf32(0.3, t);
	return ((x * (a * x + c * b) + d * e) / (x * (a * a + b) + d * f)) - e / f;
}

glm::vec3 tonemap(glm::vec3 x) {
	// tonemapping https://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting
	// linear white
	float w = 0.5;
	return tonemapPart(x) / tonemapPart(glm::vec3(w));
}

glm::vec3 sample(const TextureMap *tex, glm::vec2 uv) {
	int x = int(uv.x * tex->width) % tex->width;
	int y = int(uv.y * tex->height) % tex->height;
	auto val = tex->pixels[x + y * tex->width];
	return val;
}

enum LightingMethod {
	GGX,
	PHONG,
	BASIC,
};

glm::vec3 lightPoint(
	Intersectable *m, 
	glm::vec3 world_pos, 
	glm::vec3 normal,	
	glm::vec3 tangent,
	glm::vec2 uv,
	float mtl_idx,
	size_t index,
	Camera *cam, 
	Environment *env,
	int depth,
	LightingMethod lighting
) {
	const Material *mtl = m->material(mtl_idx);
	float roughness = mtl->roughness;
	float metallic = mtl->metallic;
	float transmission = mtl->transmission;
	float reflectance = mtl->reflectance;
	glm::vec3 albedo;
	if (mtl->albedo_map != -1) {
		const TextureMap *tex = m->texture(mtl->albedo_map);
		albedo = sample(tex, uv);
	} else {
		albedo = glm::vec3(mtl->albedo);
	};

	if (mtl->normal_map != -1) {
		const TextureMap *tex = m->texture(mtl->normal_map);
		glm::vec3 bi_tangent = glm::cross(normal, tangent);
		glm::mat3 tbn = glm::mat3(tangent, bi_tangent, normal);
		glm::vec3 n = float(2.0) * sample(tex, uv) - glm::vec3(1.0);
		n = tbn * n;
		normal = glm::normalize(n);
	}

	glm::vec3 result = glm::vec3(0.0); 
	result += albedo * env->ambient;

	for (auto &light : env->pointLights) {
		glm::vec3 light_color;
		switch (lighting) {
			case GGX:
				light_color = physically_based_point_light_calc(light, cam->getPos(), world_pos, normal, albedo, roughness, metallic);
				break;
			case PHONG:
				light_color = phong_point_light_calc(light, cam->getPos(), world_pos, normal, albedo, roughness, metallic);	
				break;
			case BASIC:
				light_color = basic_point_light_calc(light, cam->getPos(), world_pos, normal, albedo, roughness, metallic);
				break;
		}
		float shadow;
		if (index != -1) {
			shadow = light.shadowMap[index];
		} else {
			Ray shadow_ray;
			shadow_ray.src = world_pos + float(0.0001) * normal;
			shadow_ray.dir = glm::normalize(light.pos - world_pos);
			// assume not in shadow
			float density = 0.0;
			Intersection shadow_isect;
			if (m->Intersect(shadow_ray, &shadow_isect)) {
				if (shadow_isect.t < glm::length(light.pos - world_pos)) {
					density = 1.0 - m->material(shadow_isect.face.mtl)->transmission;
				}
			}
			shadow = density;
		}
		result += (float(1.0) - shadow) * light_color;
	}

	glm::vec3 ref_result = glm::vec3(0.0);
	if (reflectance != 0.0) {
		if (depth <= 0) {
			reflectance = 0.0;
		} else {
			glm::vec3 d = glm::normalize(world_pos - cam->getPos());
			Ray ref;
			ref.src = world_pos + float(0.001) * normal;
			ref.dir = glm::normalize(d - float(2.0) * (glm::dot(d, normal)) * normal);

			Intersection ref_isect;
			if (m->Intersect(ref, &ref_isect)) {
				glm::vec3 ref_pos = ref.src + ref.dir * ref_isect.t;

				glm::vec3 ref_normal = ref_isect.getNormal();
				glm::vec2 ref_uv = ref_isect.getUV();
				glm::vec3 ref_tangent = ref_isect.getTangent();

				int ref_mtl_idx = ref_isect.face.mtl;

				ref_result = lightPoint(
					m,
					ref_pos,
					ref_normal,
					ref_tangent,
					ref_uv,
					ref_mtl_idx,
					-1,
					cam,
					env,
					depth - 1,
					lighting
				);
			}
		}	
	}

	glm::vec3 trs_result = glm::vec3(0.0);
	if (transmission != 0.0) {
		if (depth <= 0) {
			transmission = 0.0;
		} else {
			glm::vec3 d = glm::normalize(world_pos - cam->getPos());
			glm::vec3 n = normal;

			float n_dot_d = glm::dot(n, d);

			float n_a;
			float n_b;
			if (n_dot_d < 0.0) {
				n_a = 1.0;
				n_b = mtl->ior;
			} else {
				n = -n;
				n_a = mtl->ior;
				n_b = 1.0;
			}

			Ray trs;
			trs.src = world_pos - float(0.001) * n;
			// https://stackoverflow.com/questions/20801561/glsl-refract-function-explanation-available
			float tmp = (n_a/n_b) * (n_a/n_b) * (float(1.0) - n_dot_d * n_dot_d);
			trs.dir = glm::normalize((n_a/n_b) * (d + d*n_dot_d) - n * sqrt(float(1.0) - tmp));

			Intersection trs_isect;
			if (m->Intersect(trs, &trs_isect)) {
				glm::vec3 trs_pos = trs.src + trs.dir * trs_isect.t;

				glm::vec3 trs_normal = trs_isect.getNormal();
				glm::vec2 trs_uv = trs_isect.getUV();
				glm::vec3 trs_tangent = trs_isect.getTangent();

				int trs_mtl_idx = trs_isect.face.mtl;

				trs_result = lightPoint(
					m,
					trs_pos,
					trs_normal,
					trs_tangent,
					trs_uv,
					trs_mtl_idx,
					-1,
					cam,
					env,
					depth - 1,
					lighting
				);
			}
		}
	}

	result = transmission * trs_result + reflectance * ref_result + (float(1.0) - transmission - reflectance) * result;
	return result;
}

inline void lightingBounded(Intersectable *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env, size_t sx, size_t sy, size_t w, size_t h, LightingMethod lighting) {
	for (size_t y = sy; y < sy+h; y++) {
		for (size_t x = sx; x < sx+w; x++) {
			size_t index = g->index(x, y);

			int mtl_idx = g->material[index];
			if (mtl_idx == -1) {
				continue;
			}

			glm::vec3 world_pos = g->world_pos[index];
			glm::vec3 normal = g->normal[index];
			glm::vec3 tangent = g->tangent[index];
			glm::vec2 uv = g->uv[index];
			float depth = g->depth[index];

			PixCoord p;
			p.x = x;
			p.y = y;
			p.depth = depth;

			glm::vec3 result = lightPoint(m, world_pos, normal, tangent, uv, mtl_idx, index, cam, env, MAX_BOUNCES, lighting);

			f->putPixel(p, glm::vec4(result.x, result.y, result.z, 1.0));
		}
	}
}

inline void lightingThreaded(Intersectable *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env, size_t numThreads, LightingMethod lighting) {
	std::vector<std::thread> threads;

	size_t h = HEIGHT / numThreads;
	size_t w = WIDTH;

	size_t y = 0;
	size_t x = 0;

	for (int i = 0; i < numThreads; i++) {
		std::thread t(lightingBounded, m, g, f, cam, env, x, y, w, h, lighting);
		threads.push_back(std::move(t));
		y += h;
	}

	for (auto &t : threads) {
		t.join();
	}
}

struct Photon {
	glm::vec3 point;
	glm::vec3 dir;
	glm::vec3 flux;
};

struct KDTreeNode {
	Photon photon;
	KDTreeNode *l;
	KDTreeNode *r;
};

class KDTree {
	public:
		KDTree(std::vector<Photon> photons) {
			this->root = buildRecursive(photons, 0);
		}

	private:
		MemoryArena arena;
		KDTreeNode *root;

		KDTreeNode *buildRecursive(
			std::vector<Photon> photons,
			int depth
		) {
			KDTreeNode *node = arena.Alloc<KDTreeNode>();
			if (photons.size() == 1) {
				node->photon = photons[0];
				node->l = nullptr;
				node->r = nullptr;
				return node;
			}

			int dim = depth % 3;

			std::vector<Photon> sorted = photons;
			std::sort(sorted.begin(), sorted.end(), [&](const Photon &a, const Photon &b) {
				return a.point[dim] < b.point[dim];
			});

			int median_idx = sorted.size() / 2;

			node->photon = sorted[median_idx];

			std::vector<Photon> left = std::vector<Photon>(sorted.begin(), sorted.begin() + median_idx);
			node->l = buildRecursive(left, depth + 1);

			std::vector<Photon> right = std::vector<Photon>(sorted.begin() + median_idx + 1, sorted.end());
			node->r = buildRecursive(right, depth + 1);

			return node;
		}
};

Ray getLightPhotonRay(const PointLight &l) {
	float u = nextRand() * 2.0 * PI;
	float v = nextRand() * PI;
	float x = sin(u) * cos(v);
	float y = cos(u) * cos(v);
	float z = sin(v);
	glm::vec3 src_offset = glm::vec3(x, y, z) * l.radius;

	// partial derivatives to get a tangent space coordinate system
	glm::vec3 ddu = glm::normalize(glm::vec3(cos(u)*cos(v), -sin(u)*cos(v), 0.0));
	glm::vec3 ddv = glm::normalize(glm::vec3(-sin(u)*sin(v), -cos(u)*sin(v), cos(v)));
	glm::vec3 n = glm::normalize(src_offset);

	float s = nextRand() * 2.0 * PI;
	float t = nextRand() * PI / 2.0;

	glm::vec3 dir = sin(s)*cos(t) * ddu + cos(s)*cos(t)*ddv + sin(t)*n;

	Ray ray;
	ray.src = l.pos + src_offset;
	// should be normalized already afaik but just to be safe
	ray.dir = glm::normalize(dir);

	return ray;
}

KDTree createPhotonMap(Intersectable *m, Environment *env, int n) {
	std::vector<float> lightBounds = std::vector<float>();
	float totalFlux = 0.0;
	for (auto &l : env->pointLights) {
		totalFlux += glm::length(l.color);
	}

	float tmp = 0.0;

	for (auto &l : env->pointLights) {
		lightBounds.push_back((glm::length(l.color) / totalFlux) + tmp);
		tmp += glm::length(l.color) / totalFlux;
	}

	std::vector<Photon> photons;

	for (int i = 0; i < n; i++) {
		// pick a light at random
		int light_idx = 0;
		float light_rng = nextRand();
		for (light_idx = 0; light_idx < env->pointLights.size(); light_idx++) {
			if (lightBounds[light_idx] >= light_rng) {
				break;
			}
		}

		Ray ray = getLightPhotonRay(env->pointLights[light_idx]);
		glm::vec3 flux = env->pointLights[light_idx].color;
		Intersection isect;
		Photon photon;

		for (int j = 0; j < MAX_BOUNCES; j++) {
			if (m->Intersect(ray, &isect)) {
				photon.dir = ray.dir;
				photon.point = ray.src + ray.dir * isect.t;
				photon.flux = flux;
				photons.push_back(photon);

				float path_rng = nextRand();
				const Material *mtl = m->material(isect.face.mtl);

				glm::vec3 normal = isect.getNormal();

				if (path_rng < mtl->reflectance) {
					// reflect the ray
					glm::vec3 d = ray.dir;
					Ray ref;
					ref.src = ray.src + ray.dir * isect.t + float(0.001) * normal;
					ref.dir = d - float(2.0) * (glm::dot(d, normal)) * normal;
					ray = ref;

					flux *= mtl->reflectance;
				} else if (path_rng < mtl->reflectance + mtl->transmission) {
					// refract the ray
					glm::vec3 d = ray.dir;
					glm::vec3 n = normal;

					float n_dot_d = glm::dot(n, d);

					float n_a;
					float n_b;
					if (n_dot_d < 0.0) {
						n_a = 1.0;
						n_b = mtl->ior;
					} else {
						n = -n;
						n_a = mtl->ior;
						n_b = 1.0;
					}

					Ray trs;
					trs.src = ray.src + ray.dir * isect.t - float(0.001) * n;
					// https://stackoverflow.com/questions/20801561/glsl-refract-function-explanation-available
					float tmp = (n_a/n_b) * (n_a/n_b) * (float(1.0) - n_dot_d * n_dot_d);
					trs.dir = glm::normalize((n_a/n_b) * (d + d*n_dot_d) - n * sqrt(float(1.0) - tmp));
					ray = trs;

					flux *= mtl->transmission;
				} else {
					// bounce the ray or absorb?
					break;
				}
			} else {
				// no intersection
				break;
			}
		}
	}

	KDTree t = KDTree(photons);
	return t;
}

inline void lighting(Intersectable *m, GeometryBuffer *g, Framebuffer *f, Camera *cam, Environment *env, LightingMethod lighting) {
	lightingBounded(m, g, f, cam, env, 0, 0, g->width, g->height, lighting);
}

struct CubicBezierCurve {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 p3;

	glm::vec3 interp(float t) {
		float omt = float(1.0) - t;

		return omt*omt*omt*p0 + float(3.0)*omt*omt*t*p1 + float(3.0)*omt*t*t*p2 + t*t*t*p3;
	}

	glm::vec3 derivative(float t) {
		float omt = 1.0 - t;
		glm::vec3 d = float(3.0)*omt*omt*(p1 - p0) + float(6.0)*omt*t*(p2-p1) + float(3.0)*t*t*(p3-p2);
		return glm::normalize(d);
	}
};

int main(int argc, char *argv[]) {
	glm::mat4 model = glm::mat4(
		0.5, 0.0, 0.0, 0.0,
		0.0, 0.5, 0.0, 0.0,
		0.0, 0.0, 0.5, 0.0,
		0.0, 0.0, 0.0, 1.0
	);
	Mesh m = Mesh("scene.obj", model);
	BVH bvh = BVH(&m, 5);

	// float theta = 0.1;
	Camera cam = Camera();

	PointLight light;
	light.pos = glm::vec3(0.0, 1.0, 1.0);
	light.falloff = 0.5;
	light.color = glm::vec3(2.0);
	light.radius = 0.05;
	light.shadowMap = std::vector<float>(WIDTH * HEIGHT, 0.0);

	float ambient = 0.05;
	std::vector<PointLight> pointLights = std::vector<PointLight>();
	pointLights.push_back(light);
	Environment env = Environment(pointLights, ambient);

	MyWindow w = MyWindow(WIDTH, HEIGHT, false);
	GeometryBuffer g = GeometryBuffer(WIDTH, HEIGHT);

	float fps = 24.0;
	float t = 0.0;
	float dt = 1.0 / fps;

	int frame_num = 1036;

	CubicBezierCurve c;
	float theta = 25.39;
	c.p0 = float(5.0) * glm::vec3(cos(-theta + (PI/2)), 0.0, sin(-theta + (PI/2)));
// c.p0 = glm::vec3(1.011145830154419, 0.0, 4.960288047790527);
c.p1 = glm::vec3(0.011188507080078125, 0.0, 4.951050758361816);
c.p2 = glm::vec3(-1.507867455482483, -0.7425732016563416, 2.3361434936523438);
c.p3 = glm::vec3(-1.1663074493408203, 0.08422451466321945, 1.4649369716644287);


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

		// float theta = (t / 5.0) * 2.0 * PI;
		// float theta = 25.39;
		// // theta = 0.00001;
		// cam = Camera(theta);
		// glm::vec3 p = c.interp(0.0);
		// glm::vec3 d = c.derivative(0.0);
		// cam = Camera(p, d);

		// glm::vec3 p = glm::vec3(1.011145830154419, 0.0, 4.960288047790527);
		

		// glm::vec3 p = c.interp(t / 10.0);
		// std::cout << p.x << " " << p.y << " " << p.z << std::endl;
		// glm::vec3 d = c.derivative(t / 10.0);
		// std::cout << d.x << " " << d.y << " " << d.z << std::endl;
		// float pitch = asinf32(d.y);
		// float yaw = acosf32(d.z / cosf32(pitch));

		// glm::vec3 p = c.p3;//glm::vec3(-0.8470036387443542, 0.716678261756897, 0.4659724831581116);
		if(t > 10.0) {
			break;
			// t = 0.0;
		}

		glm::vec3 p = c.p3;//c.interp(t / 10.0);
		glm::vec3 d = glm::normalize(p);
		float pitch = -asinf32(d.y);
		float yaw = acosf32(d.z / cosf32(pitch));
		if (d.x < 0.0) {
			yaw = -yaw;
		}

		std::cout << p.x << " " << p.y << " " << p.z << std::endl;
		std::cout << pitch << std::endl;
		std::cout << yaw << std::endl;

		cam = Camera(p, pitch, yaw);

		t += dt;

		const auto before = std::chrono::system_clock::now();

		

		// if (t > 15.0) {
			std::cout << "trace" << std::endl;
			traceThreaded(&bvh, &cam, &g, 8);
			std::cout << "shadow" << std::endl;
			shadowThreaded(&bvh, &g, &env, 8);
			std::cout << "lighting" << std::endl;
			lightingThreaded(&bvh, &g, &w, &cam, &env, 8, GGX);
			// drawMeshWireframe(&m, &cam, &w);
		// } else if (t > 10.0) {
			// std::cout << "trace" << std::endl;
			// traceThreaded(&bvh, &cam, &g, 8);
			// std::cout << "shadow" << std::endl;
			// shadowThreaded(&bvh, &g, &env, 8);
			// std::cout << "lighting" << std::endl;
			// lightingThreaded(&bvh, &g, &w, &cam, &env, 8, BASIC);
		// } else if (t > 5.0) {
		// 	drawMesh(&m, &cam, &w);
		// } else {
		// 	drawMeshWireframe(&m, &cam, &w);
		// }

		const std::chrono::duration<double, std::milli> duration = std::chrono::system_clock::now() - before;
		std::cout << duration.count() << std::endl;

		w.window.renderFrame();

		std::cout << "render" << std::endl;
		std::cout << std::endl;

		std::ostringstream ss;
		ss << "output/" << std::setw(5) << std::setfill('0') << frame_num  << ".ppm";
		w.window.savePPM(ss.str());
		frame_num++;
	}
}
