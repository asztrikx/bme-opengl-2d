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

#define dbg if(true)
float massUnit = 1.6735575e-27;
float chargeUnit = 1.60218e-19;
float distanceUnit = 0.5*1e-2;

int massRange = 2;
int chargeAbsRange = 2;
float atomRadius = 3;
float atomRadiusEps = atomRadius * 1.5f;
int dtMs = 10;
float dt = dtMs/1000.0f;
float dragConstant = 100e-27;

int randBetween(int min, int max) {
	return min + (std::rand() % (max - min + 1));
}

float randFloatBetween(int min, int max) {
	return (float) std::rand() / RAND_MAX * (max - min) + min;
}

class Camera2D {
	vec2 position = vec2(0, 0);
	vec2 size = vec2(45,45);

  public:
	mat4 V() { return TranslateMatrix(-position); }
	mat4 P() { return ScaleMatrix(vec3(2/size.x, 2/size.y, 0)); }

	void Pan(vec2 translate) { position = position + translate*size; }
	void Reset() { position = vec2(0,0); }
};

GPUProgram gpuProgram;
Camera2D camera;

class Circle {
	const int tesselationCount = 50;
	const int count = tesselationCount + 1;
	unsigned int vao;

  public:
	void Create() {
		glGenVertexArrays(1, &vao);
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

	void Draw(mat4 MVP, vec3 color) {
		glBindVertexArray(vao);

		gpuProgram.setUniform(MVP, "MVP");
		gpuProgram.setUniform(color, "color");

		glDrawArrays(GL_TRIANGLE_FAN, 0, count);
	}
};

Circle circle;

struct Atom {
	vec2 position;
	float m, q, radius;
	vec3 color;

	// TODO take radius out
	Atom(float radius) : radius(radius) { }

	mat4 M() { return ScaleMatrix(vec2(radius, radius)) * TranslateMatrix(position); }

	void Draw() {
		mat4 mvp = M() * camera.V() * camera.P();
		circle.Draw(mvp, color);
	}
};

class GraphCreator {
	float radius, radiusEps;
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

	bool pointCrossesAny(vec2 point) {
		float diameter = 2 * radius;
		float minDistance = diameter + radiusEps;

		for (int j = 0; j < points.size(); j++) {
			vec2 d = point - points[j];
			if (length(d) < minDistance) {
				return true;
			}
		}

		for (auto edge: edges) {
			if (edgeCrossesCircle(edge.first, edge.second, point)) {
				return true;
			}
		}

		return false;
	}

  public:
	std::vector<vec2> points;
	std::vector<std::pair<int, int>> edges;

	GraphCreator(float radius, float radiusEps) : radius(radius), radiusEps(radiusEps) {
		int size = randBetween(2, 8);
		vec2 min(-rectSize/2, -rectSize/2);
		vec2 max(rectSize/2, rectSize/2);

		for (int i = 0; i < size; i++) {
			vec2 point;
			do {
				point = vec2(randFloatBetween(min.x, max.x), randFloatBetween(min.y, max.y));
			} while(pointCrossesAny(point));
			points.push_back(point);

			// edge creation
			if(i == 0) { continue; } // <= rand(0, -1)
			int index;
			do{
				index = randBetween(0, i-1);
			} while(edgeCrossesEdgeAny(points[i], points[index]) || edgeCrossesCircleAny(i, index));
			edges.push_back(std::make_pair(i, index));
		}
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

class Molecule {
	std::vector<std::pair<int,int>> edges;
	int rectSize = 100.0f;
	unsigned int vao;
	unsigned int vbo;
	int edgeTesselation = 100;
	vec2 position = vec2(0, 0);

  public:
	vec2 v = vec2(0, 0);
	float alpha = 0;
	float omega = 0;
	std::vector<Atom> atoms;
	float angularMass = 0;
	vec2 getCentroid() { return position; }

	void addChanges(MoleculeChange moleculeChange) {
		alpha += moleculeChange.alpha;
		omega += moleculeChange.omega; // TODO overflow
		position = position + moleculeChange.position;
		v = v + moleculeChange.v;

		for (Atom& atom: atoms) {
			atom.position = atom.position + moleculeChange.position;
			vec4 p(atom.position.x, atom.position.y, 0, 1);
			p = p * MAtom(alpha);
			atom.position = vec2(p.x, p.y);
		}
	}

	Molecule() {
		GraphCreator graphCreator(atomRadius, atomRadiusEps);
		std::vector<vec2> points = graphCreator.points;
		atoms.resize(points.size(), Atom(atomRadius));
		for (size_t i = 0; i < atoms.size(); i++) {
			atoms[i].position = points[i];
		}
		edges = graphCreator.edges;

		// tesselation
		std::vector<vec2> edgePoints;
		for (size_t i = 0; i < edges.size(); i++) {
			vec2 a = atoms[edges[i].first].position;
			vec2 b = atoms[edges[i].second].position;

			for (int j = 0; j < edgeTesselation; j++) {
				float t = (float) j / (edgeTesselation - 1);
				vec2 p = a*t + b*(1-t);
				edgePoints.push_back(p);
			}
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

	void openGlInit(std::vector<vec2> &edgePoints) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
		glBufferData(GL_ARRAY_BUFFER, edgePoints.size()*sizeof(vec2), &edgePoints[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	}

	~Molecule() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}

	mat4 M() { return RotationMatrix(alpha, vec3(0,0,1)) * TranslateMatrix(position); }

	mat4 MAtom(float a) { return TranslateMatrix(-getCentroid()) * RotationMatrix(a, vec3(0,0,1)) * TranslateMatrix(getCentroid()); }

	void Draw() {
		glBindVertexArray(vao);
		glLineWidth(2.0f);

		mat4 mvp = M() * camera.V() * camera.P();
		gpuProgram.setUniform(mvp, "MVP");
		gpuProgram.setUniform(vec3(1,1,1), "color");

		for (int i = 0; i < edges.size(); i++) {
			glDrawArrays(GL_LINE_STRIP, i*edgeTesselation, edgeTesselation);
		}

		for(Atom atom : atoms) {
			atom.Draw();
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

	circle.Create();
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
		case 's': camera.Pan(vec2(-panUnit,0)); break;
		case 'd': camera.Pan(vec2(panUnit,0)); break;
		case 'x': camera.Pan(vec2(0,-panUnit)); break;
		case 'e': camera.Pan(vec2(0,panUnit)); break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

MoleculeChange physics(Molecule &reference, Molecule &actor) {
	float sumM = 0;
	vec2 sumFc_move(0, 0);
	float summ = 0;
	for (Atom refAtom: reference.atoms) {
		vec2 sumFc(0,0);
		vec2 r = (refAtom.position - reference.getCentroid())*distanceUnit;
		for (Atom actorAtom: actor.atoms) {
			// Fc
			float k = 2*8.9875517923e9;
			vec2 d = (refAtom.position - actorAtom.position) * distanceUnit; // TODO really small distance
			//dbg printf("q1, q2: %le %le\n", refAtom.q, actorAtom.q);
			vec2 Fc = k * (refAtom.q*actorAtom.q) / length(d) * normalize(d);
			sumFc = sumFc + Fc;
		}
		//dbg printf("sum force: %le %le %le\n", sumFc.x, sumFc.y, reference.omega);

		vec2 Fc_k = dot(sumFc, normalize(r)) * normalize(r);
		vec2 Fc_move = sumFc - Fc_k;

		//dbg printf("Fck force: %le %le %le\n", Fc_k.x, Fc_k.y, reference.omega);
		// Fd
		vec3 v_k = cross(vec3(0,0,reference.omega), vec3(r.x, r.y, 0));
		vec2 Fd_k = -dragConstant * vec2(v_k.x, v_k.y);

		//dbg printf("Fdk force: %le %le %le\n", Fd_k.x, Fd_k.y, reference.omega);

		vec2 F_k = Fc_k + Fd_k;
		float M = cross(vec3(r.x, r.y, 0), vec3(F_k.x, F_k.y, 0)).z;
		
		sumM += M;
		summ += refAtom.m;
		sumFc_move = sumFc_move + Fc_move;
	}
	vec2 Fd_move = -dragConstant * reference.v;
	vec2 F_move = sumFc_move + Fd_move;

	MoleculeChange moleculeChange;
	moleculeChange.v = F_move/summ * dt;
	moleculeChange.position = reference.v * dt / distanceUnit;
	moleculeChange.omega = sumM/reference.angularMass * dt;
	moleculeChange.alpha = reference.omega * dt;

	dbg printf("v: %le %le\n", moleculeChange.v.x/distanceUnit, moleculeChange.v.y/distanceUnit);
	dbg printf("p: %le %le\n", moleculeChange.position.x, moleculeChange.position.y);
	dbg printf("beta: %le\n", moleculeChange.omega);
	dbg printf("alpha: %le\n", moleculeChange.alpha);

	return moleculeChange;
}

long lastTime = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);

	for (long t = lastTime+dtMs; t <= time; t += dtMs) {
		lastTime = t;

		std::vector<MoleculeChange> moleculeChanges(molecules.size());
		for (int i = 0; i < molecules.size(); i++) {
			for (int j = 0; j < molecules.size(); j++) {
				if(i == j) {
					continue;
				}
				dbg printf("------m%d\n", i);
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
