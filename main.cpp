#include "data.h"
#include "gravity_calculations.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "Usage: nbody <NUM_PARTICLES> [OUTPUT]" << endl;
    return 1;
  }

  BoundingBox bb(0, 0, 0, 100000, 100000, 100000);
  Particles particles = generate_random_particles(atoi(argv[1]), 0, 100000);
  OctreeNode *parent = new OctreeNode(particles, bb);

  init_semaphore(16);
  make_tree_bounded(static_cast<void *>(parent));
  deinit_semaphore();
  // make_tree_serial(parent);
  // make_tree_unbounded(static_cast<void *>(parent));
  // make_tree_openmp(parent);

  if (argv[2]) {
    std::string filename = "output.txt";
    DataOutput output(filename);
    output.add_datapoints(particles);
    output.add_octree_bounding_boxes(parent);
  }

  return 0;
}
