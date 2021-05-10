#include "data.h"
#include "gravity_calculations.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "Usage: nbody <NUM_PARTICLES> [OUTPUT] <MODE> [Bounded # Threads]" << endl;
    return 1;
  }

  BoundingBox bb(0, 0, 0, 100000, 100000, 100000);
  Particles particles = generate_random_particles(atoi(argv[1]), 0, 100000);
  OctreeNode *parent = new OctreeNode(particles, bb);

  // Mode 0 is serial
  if (argv[3] == 0)
    make_tree_serial(parent);

  // Mode 1 is unbounded parallel
  else if (argv[3] == 1)
    make_tree_unbounded(static_cast<void *>(parent));

  // Mode 2 is bounded parallel
  else if (argv[3] == 2) {
    if (argc < 5) {
      cout << "Invalid number of threads" << endl;
      return 1;
    }
    init_semaphore(argv[4]);
    make_tree_bounded(static_cast<void *>(parent));
    deinit_semaphore();
  }

  // Mode 3 is OpenMP
  else if (argv[3] == 3)
    make_tree_openmp(parent);

  // Default - Invalid Arguments
  else
    cout << "Invalid Mode Argument" << endl;

  // Print data to file
  if (argv[2]) {
    std::string filename = "output.txt";
    DataOutput output(filename);
    output.add_datapoints(particles);
    output.add_octree_bounding_boxes(parent);
  }

  return 0;
}
