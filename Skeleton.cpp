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
	layout(location = 0) in vec2 v;

	void main() {
		vec4 v2 = vec4(v.x, v.y, 0, 1) * MVP;
		float w = pow(v2.x*v2.x + v2.y*v2.y + 1, 0.5);
		gl_Position = vec4(v2.x/(w + 1), v2.y/(w + 1), 0, 1);
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 outColor;

	void main() {
		outColor = vec4(color.xyz, 1);
	}
)";

float massUnit = 1.6735575e-27;
float chargeUnit = 1.60218e-19;
float distanceUnit = 1e-2;

int massRange = 3;
int chargeAbsRange = 3;
float atomRadius = 3;
float atomRadiusEps = atomRadius * 1.5f;
float dtMs = 10;
float dt = dtMs/1000;
float dragConstant = 8e-27;

// TODO randFloatBetween
int randBetween(int min, int max) {
	return min + (std::rand() % (max - min + 1));
}

class Camera2D {
	vec2 position = vec2(0, 0);
	vec2 size = vec2(100, 100);

  public:
	mat4 V() { return TranslateMatrix(-position); }
	mat4 P() { return ScaleMatrix(vec3(2/size.x, 2/size.y, 0)); }

	void Pan(vec2 translate) { position = position + translate*size; }
	// TODO delete
	void Zoom(float scalar) {size = size * scalar; }
	void Reset() { size = vec2(100,100); position = vec2(0,0); }
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

	static void Draw(mat4 MVP, vec3 color) {
		glBindVertexArray(vao);

		gpuProgram.setUniform(MVP, "MVP");
		gpuProgram.setUniform(color, "color");

		glDrawArrays(GL_TRIANGLE_FAN, 0, count);
	}
};

struct Atom {
	vec2 position;
	float m, q, radius;
	vec3 color;

	// TODO take radius out
	Atom(float radius) : radius(radius) { }

	mat4 M() { return ScaleMatrix(vec2(radius, radius)) * TranslateMatrix(position); }

	void Draw(mat4 T) {
		mat4 mvp = M() * T * camera.V() * camera.P();
		Circle::Draw(mvp, color);
	}
};
// TODO float rand / RAND_MAX, custom pair
class GraphCreator {
	float radius, radiusEps;
	std::vector<int> groups;
	int uniqueGroups;
	std::vector<vec2> points;
	std::vector<std::pair<int, int>> edges;
	float rectSize = 50.0f;

	int direction(vec2 base, vec2 from, vec2 to) {
		from = from - base;
		to = to - base;
		float area = from.y*to.x - from.x * to.y;
		if (area < 0) { return -1; }
		if (area > 0) { return 1; }
		return 0;
	}

	bool edgeCrossesEdge(vec2 a, vec2 b, vec2 c, vec2 d) {
		return direction(a, b, c) * direction(a, b, d) < 0 &&
				direction(c, d, a) * direction(c, d, b) < 0;
	}

	bool edgeCrossesEdgeAny(vec2 a, vec2 b) {
		for (auto otherEdge : edges) {
			if (edgeCrossesEdge(a, b, points[otherEdge.first], points[otherEdge.second])) {
				return true;
			}
		}
		return false;
	}

	bool edgeCrossesCircle(vec2 a, vec2 b, vec2 p) {
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
		return fabs(distance) <= radius + radiusEps;
	}

	bool edgeCrossesCircleAny(int aIndex, int bIndex) {
		for(int i = 0; i < points.size(); i++) {
			if (i == aIndex || i == bIndex) {
				continue;
			}
			if (edgeCrossesCircle(points[aIndex], points[bIndex], points[i])) {
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
	GraphCreator(float radius, float radiusEps) : radius(radius), radiusEps(radiusEps) {}

	std::vector<std::pair<int, int>> getEdges() {
		int n = points.size();
		groups.resize(n);
		uniqueGroups = n;
		for (int i = 0; i < n; i++) {
			groups[i] = i;
		}

		std::vector<std::pair<int, int>> allEdges;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				allEdges.push_back(std::make_pair(i,j));
			}
		}
		
		while (uniqueGroups != 1) {
			int index = randBetween(0, allEdges.size()-1);
			std::pair<int, int> edge = allEdges[index];
			vec2 a = points[edge.first];
			vec2 b = points[edge.second];

			if(edgeCrossesEdgeAny(a, b) || edgeCrossesCircleAny(edge.first, edge.second)) {
				continue;
			}

			int rootA = find(edge.first);
			int rootB = find(edge.second);
			if (rootA == rootB) {
				continue;
			}

			groups[rootA] = groups[rootB];
			uniqueGroups--;

			allEdges.erase(allEdges.begin() + index);
			edges.push_back(edge);
		}

		return edges;
	}

	std::vector<vec2> getPoints() {
		points.resize(randBetween(2, 8));
		vec2 min(-rectSize/2, -rectSize/2);
		vec2 max(rectSize/2, rectSize/2);
		float diameter = 2 * radius;
		float minDistance = diameter + radiusEps;

		for (size_t i = 0; i < points.size(); i++) {
			bool good;
			do {
				good = true;
				points[i] = vec2(randBetween(min.x, max.x), randBetween(min.y, max.y));

				for (size_t j = 0; j < i; j++) {
					vec2 distanceVec = points[i] - points[j];
					float distance = dot(distanceVec, distanceVec);
					if (distance < minDistance * minDistance) {
						good = false;
						break;
					}
				}
			} while(!good);
		}

		return points;
	}
};

struct MoleculeChange {
	vec2 v = vec2(0,0), position = vec2(0,0);
	float omega=0, alpha=0;

	MoleculeChange() {}

	MoleculeChange operator+(MoleculeChange other) {
		MoleculeChange moleculeChange;
		moleculeChange.position = position + other.position;
		moleculeChange.v = v + other.v;
		moleculeChange.alpha = alpha + other.alpha;
		moleculeChange.omega = omega + other.omega;
		return moleculeChange;
	}
};

// TODO do not place m1's atom over m2's
// TODO tessellate edges
class Molecule {
	std::vector<std::pair<int,int>> edges;
	int rectSize = 100.0f;
	unsigned int vao;
	unsigned int vbo;
	vec2 position;

  public:
	vec2 v;
	float alpha;
	float omega;
	std::vector<Atom> atoms;
	float angularMass;
	vec2 getCentroid() { return position; }

	Molecule() {
		GraphCreator graphCreator(atomRadius, atomRadiusEps);
		auto points = graphCreator.getPoints();
		atoms.resize(points.size(), Atom(atomRadius));
		for (size_t i = 0; i < atoms.size(); i++) {
			atoms[i].position = points[i];
		}
		edges = graphCreator.getEdges();

		// tesselation
		std::vector<vec2> edgePoints(edges.size()*2);
		for (size_t i = 0; i < edges.size(); i++) {
			edgePoints[2*i] = atoms[edges[i].first].position;
			edgePoints[2*i+1] = atoms[edges[i].second].position; 
		}

		// rand m, q
		int sumCharge = 0;
		for (Atom& atom: atoms) {
			atom.m = randBetween(1, massRange);
			atom.q = randBetween(-chargeAbsRange, chargeAbsRange);
			sumCharge += atom.q;
		}

		if (sumCharge != 0) {
			int signal = sumCharge < 0 ? -1 : 1;
			sumCharge = abs(sumCharge);
			for (int i = 1; i <= sumCharge; i++) {
				do {
					int index = randBetween(0, atoms.size()-1);
					if (atoms[index].q != chargeAbsRange * -signal) {
						atoms[index].q -= signal;
						break;
					}
				} while (true);
			}
		}

		// color, chargeUnit
		for (Atom& atom: atoms) {
			float intensity = 1.0f * fabs(atom.q / chargeAbsRange);
			vec3 endColor;
			if (atom.q < 0) {
				endColor = vec3(0,0,1);
			} else {
				endColor = vec3(1,0,0);
			}
			atom.color = vec3(0,0,0) * (1-intensity) + endColor * intensity;
			atom.q = atom.q * chargeUnit;
		}
        
        // balancepoint
		vec2 centroid(0, 0);
		float sumMass = 0;
		for (Atom &atom : atoms) {
			centroid = centroid + atom.m * atom.position;
			sumMass += atom.m;
			atom.m *= massUnit;
		}
		centroid = centroid / sumMass;

		// balancepoint -> origo
		for (vec2 &edgePoint: edgePoints) {
			edgePoint = edgePoint - centroid;
		}
		for (Atom &atom: atoms){
			atom.position = atom.position - centroid;
		}

		// angular mass
		angularMass = 0;
		for (Atom atom: atoms) {
			angularMass = atom.m * dot(atom.position,atom.position) * distanceUnit * distanceUnit;
			angularMass *= 10;
		}

		// random position
		int x = randBetween(-rectSize/2, rectSize/2);
		int y = randBetween(-rectSize/2, rectSize/2);
		position = vec2(x,y);
		for (Atom& atom: atoms) {
			atom.position = atom.position + position;
		}
		
		openGlInit(edgePoints);
	}

	void addChanges(MoleculeChange moleculeChange) {
		alpha += moleculeChange.alpha;
		omega += moleculeChange.omega; // TODO overflow
		position = position + moleculeChange.position;
		v = v + moleculeChange.v;

		for (Atom& atom: atoms) {
			atom.position = atom.position + moleculeChange.position;
		}
	}

	void openGlInit(std::vector<vec2> &edgePoints) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
		// SIZEOF(FLOAT)
		glBufferData(GL_ARRAY_BUFFER, edgePoints.size()*sizeof(vec2), &edgePoints[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		// SIZEOF(FLOAT)
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	}

	~Molecule() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}

	mat4 M() { return RotationMatrix(alpha, vec3(0,0,1)) * TranslateMatrix(position); }

	void Draw() {
		glBindVertexArray(vao);
		glLineWidth(2.0f);

		mat4 mvp = M() * camera.V() * camera.P();
		gpuProgram.setUniform(mvp, "MVP");
		gpuProgram.setUniform(vec3(1,1,1), "color");
		glDrawArrays(GL_LINES, 0, 2*edges.size());

		for(Atom atom : atoms) {
			atom.Draw(TranslateMatrix(-getCentroid()) * RotationMatrix(alpha, vec3(0,0,1)) * TranslateMatrix(getCentroid()));
		}
	}
};

std::vector<Molecule*> molecules;

void restart() {
	for (int i = 0; i < molecules.size(); i++) {
		if (molecules[i] != nullptr) {
			delete molecules[i];
		}
		molecules[i] = new Molecule();
	}
	camera.Reset();
}

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	Circle::Create();
	for (int i = 1; i <= 2; i++) {
		molecules.push_back(nullptr);
	}
	
	restart();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0.5, 0.5, 0.5, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for(Molecule *molecule: molecules){
		molecule->Draw();
	}

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	float panUnit = 0.1;

	switch (key){
		case ' ': restart(); break;
		case 'a': camera.Pan(vec2(-panUnit,0)); break;
		case 'd': camera.Pan(vec2(panUnit,0)); break;
		case 's': camera.Pan(vec2(0,-panUnit)); break;
		case 'w': camera.Pan(vec2(0,panUnit)); break;
		case 'z': camera.Zoom(0.9f); break;
		case 'Z': camera.Zoom(1.1f); break;
	}
	glutPostRedisplay();
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

MoleculeChange physics(Molecule &reference, Molecule &actor) {
	vec2 moleculaA(0,0);
	float moleculaB = 0;
	for (Atom refAtom: reference.atoms) {
		vec2 atomF(0,0);
		for (Atom actorAtom: actor.atoms) {
			// Fc
			float k = 8.9875517923e9;
			vec2 d = (refAtom.position - actorAtom.position) * distanceUnit;
			vec2 idk = 1/dot(d,d) * normalize(d);
			vec2 Fc = k*(refAtom.q*actorAtom.q) * idk;

			vec2 r = (refAtom.position - reference.getCentroid())*distanceUnit;

			// Fd
			vec3 tmp = cross(vec3(0,0,reference.omega), vec3(r.x, r.y, 0));
			vec2 v = reference.v + vec2(tmp.x, tmp.y);
			vec2 Fd = -dragConstant * v;

			vec2 F = Fc+Fd;

			float M = cross(vec3(r.x, r.y, 0), vec3(F.x, F.y, 0)).z;
			moleculaB += M/reference.angularMass;
			atomF = atomF + F;
		}

		moleculaA = moleculaA + atomF/refAtom.m;
	}

	MoleculeChange moleculeChange;
	moleculeChange.position = reference.v * dt / distanceUnit;
	moleculeChange.v = moleculaA * dt;
	moleculeChange.alpha = reference.omega * dt;
	moleculeChange.omega = moleculaB * dt;

	return moleculeChange;
}

float lastTime = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);

	for (float t = lastTime+dtMs; t <= time; t += dtMs) {
		lastTime = t;

		std::vector<MoleculeChange> moleculeChanges(molecules.size());
		for (int i = 0; i < molecules.size(); i++) {
			for (int j = 0; j < molecules.size(); j++) {
				if(i == j) {
					continue;
				}
				moleculeChanges[i] = moleculeChanges[i] + physics(*molecules[i], *molecules[j]);
			}
		}

		// Apply changes
		for (int i = 0; i < molecules.size(); i++) {
			molecules[i]->addChanges(moleculeChanges[i]);
		}
	}
	glutPostRedisplay();
}
