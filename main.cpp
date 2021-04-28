#include "data.h"
#include "gravity_calculations.h"

#include <boost/program_options.hpp>
#include <iostream>

using namespace boost::program_options;

int main(int argc, char **argv) {
  BoundingBox bb(0, 0, 0, 100, 100, 100);
  Particles particles = generate_random_particles(100, 0, 100);
  OctreeNode *parent = new OctreeNode(particles, bb);

  std::cout << "Completed particle generation!" << std::endl;

  // make_tree(static_cast<void *>(parent));
  make_tree_openmp(parent);

  std::string filename = "output.txt";
  DataOutput output(filename);
  output.add_datapoints(particles);
  output.add_octree_bounding_boxes(parent);

  return 0;
}
