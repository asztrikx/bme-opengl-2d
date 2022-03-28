//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Vörös Asztrik
// Neptun : WYZJ90
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec4 color;
	out vec4 outColor;

	void main() {
		outColor = color;
	}
)";

int randBetween(int min, int max) {
	return min + (std::rand() % (max - min + 1));
}

class Camera2D {
	vec2 position = vec2(0, 0);
	vec2 size = vec2(100, 100);

  public:
	mat4 V() { return TranslateMatrix(position); }
	mat4 P() { return ScaleMatrix(vec3(2/size.x, 2/size.y, 0)); }
};

GPUProgram gpuProgram;
Camera2D camera;

class Circle {
	static const int tesselationCount = 50;
	static const int count = tesselationCount + 1;
	inline static unsigned int vao;

  public:
	static void Create() {
		glGenVertexArrays(1, &Circle::vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		std::vector<vec2> points(count);
		points[0] = vec2(0,0);
		for (size_t i = 0; i < tesselationCount; i++) {
			float angle = 2*M_PI * i/(tesselationCount - 1);
			points[i+1] = vec2(cosf(angle), sinf(angle));
		}
		glBufferData(GL_ARRAY_BUFFER, points.size()*sizeof(vec2), &points[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), nullptr);
	}

	static void Draw(mat4 MVP, vec4 color) {
		glBindVertexArray(vao);

		gpuProgram.setUniform(MVP, "MVP");
		gpuProgram.setUniform(color, "color");

		glDrawArrays(GL_TRIANGLE_FAN, 0, count);
	}
};

struct Atom {
	vec2 position;
	float m, q, radius;

	Atom(float radius) : radius(radius) { }

	mat4 M() { return TranslateMatrix(position); }

	void Draw() {
		mat4 mvp = M() * camera.V() * camera.P();
		vec4 color = vec4(0,0,0, 1);
		if (q < 0) {
			color.z = 1;
		} else {
			color.x = 1;
		}
		
		Circle::Draw(mvp, color);
	}
};

class GraphCreator {	
	float radius;
	std::vector<int> groups;
	int uniqueGroups;
	std::vector<vec2> points;
	std::vector<std::pair<int, int>> edges;

	int direction(vec2 base, vec2 from, vec2 to) {
		from = from - base;
		to = to - base;
		float area = from.y*to.x - from.x * to.y;
		if (area < 0) { return -1; }
		if (area > 0) { return 1; }
		return 0;
	}

	bool crosses(vec2 a, vec2 b, vec2 c, vec2 d) {
		return direction(a, b, c) * direction(a, b, d) < 0 &&
				direction(c, d, a) * direction(c, d, b) < 0;
	}

	bool crossesAny(vec2 a, vec2 b) {
		for (auto otherEdge : edges) {
			if (crosses(a, b, otherEdge.first, otherEdge.second)) {
				return true;
			}
		}
		return false;
	}

	bool pointNear(vec2 a, vec2 b, vec2 p) {
		vec2 v1 = a-b;
		v1 = vec2(v1.y, -v1.x);
		vec2 v2 = b-a;
		v2 = vec2(-v2.y, v2.x);
		if (direction(vec2(0,0), v1, p-b) * direction(vec2(0,0), v2, p-a) > 0) {
			return false;
		}

		vec2 v = b-a;
		vec2 n = vec2(-v.y, v.x);
		vec2 n0 = normalize(n);
		float distance = dot(p-a, n0);
		return distance <= radius;
	}

	bool pointNearAny(int aIndex, int bIndex) {
		for(int i = 0; i < points.size(); i++) {
			if (i == aIndex || i == bIndex) {
				continue;
			}
			if (pointNear(points[aIndex], points[bIndex], points[i])) {
				return true;
			}
		}
		return false;
	}

	int find(int i) {
		if (groups[i] == i) {
			return i;
		}
		return groups[i] = find(groups[i]);
	}

  public:
	GraphCreator(float radius) : radius(radius) {}

	std::vector<std::pair<int, int>> getEdges() {
		int n = points.size();
		groups.resize(n);
		uniqueGroups = n;
		for (size_t i = 0; i < n; i++) {
			groups[i] = i;
		}

		std::vector<std::pair<int, int>> alledges;
		for (size_t i = 0; i < n; i++) {
			for (size_t j = i + 1; j < n; j++){
				alledges.push_back(std::make_pair(i,j));
			}
		}
		
		while (uniqueGroups != 1) {
			int index = randBetween(0, alledges.size()-1);
			std::pair<int, int> edge = alledges[index];
			vec2 a = points[edge.first];
			vec2 b = points[edge.second];

			if(crossesAny(a, b) || pointNearAny(edge.first, edge.second)) {
				continue;
			}

			if (find(edge.first) == find(edge.second)) {
				continue;
			}

			groups[edge.first] = groups[edge.second];
			uniqueGroups--;

			alledges.erase(alledges.begin() + index);
			edges.push_back(edge);
		}

		return edges;
	}

	std::vector<vec2> getPoints() {
		points.resize(randBetween(2, 8));
		vec2 min(-25, -25);
		vec2 max(25, 25);

		for (size_t i = 0; i < points.size(); i++) {
			bool good;
			do {
				points[i] = vec2(randBetween(min.x, max.x), randBetween(min.y, max.y));

				for (size_t j = 0; j < i; j++) {
					vec2 distanceVec = points[i] - points[j];
					float distance = dot(distanceVec, distanceVec);
					if (distance < radius * radius) {
						good = false;
						break;
					}
				}
				good = true;
			} while(!good);
		}

		return points;
	}
};

class Molecule {
	vec2 position;
	std::vector<Atom> atoms;
	std::vector<std::pair<int,int>> edges;
	unsigned int vao;
	float atomRadius = 10;

  public:
	Molecule() {
		GraphCreator graphCreator(atomRadius);
		auto points = graphCreator.getPoints();
		atoms.resize(points.size(), Atom(atomRadius));
		for (size_t i = 0; i < atoms.size(); i++) {
			atoms[i].position = points[i];
		}
		edges = graphCreator.getEdges();

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		std::vector<vec2> edgePoints(edges.size()*2);
		for (size_t i = 0; i < edges.size(); i++) {
			edgePoints[2*i] = edges[i].first; 
			edgePoints[2*i+1] = edges[i].second; 
		}
		
		// SIZEOF(FLOAT)
		glBufferData(GL_ARRAY_BUFFER, edgePoints.size()*2*sizeof(float), &edgePoints[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		// SIZEOF(FLOAT)
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
	}

	mat4 M() { return TranslateMatrix(vec2(0,0)); }

	void Draw() {
		glLineWidth(2.0f);
		glBindVertexArray(vao);

		mat4 mvp = M() * camera.V() * camera.P();
		gpuProgram.setUniform(mvp, "MVP");
		glDrawArrays(GL_LINES, 0, 2*edges.size());

		for(Atom atom : atoms) { atom.Draw(); }
	}
};

Molecule* molecule1;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	Circle::Create();
	molecule1 = new Molecule();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0.5, 0.5, 0.5, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	molecule1->Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY) {

	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
}
