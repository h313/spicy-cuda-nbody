#include "data.h"
#include "gravity_calculations.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Usage: nbody <NUM_PARTICLES>" << endl;
    return 1;
  }

  BoundingBox bb(0, 0, 0, 100, 100, 100);
  Particles particles = generate_random_particles(atoi(argv[1]), 0, 100);
  OctreeNode *parent = new OctreeNode(particles, bb);

  cout << "Completed particle generation!" << endl;

  // make_tree(static_cast<void *>(parent));
  make_tree_openmp(parent);

  std::string filename = "output.txt";
  DataOutput output(filename);
  output.add_datapoints(particles);
  output.add_octree_bounding_boxes(parent);

  return 0;
}
