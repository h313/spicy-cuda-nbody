#include "gravity_calculations.h"
#include "generator.h"

#include <iostream>

int main(int argc, char **argv) {
  BoundingBox bb(0, 0, 0, 100, 100, 100);
  OctreeNode *parent =
      new OctreeNode(generate_random_particles(100, 0, 100), bb);

  std::cout << "Completed particle generation!" << std::endl;

  make_tree(static_cast<void *>(parent));
  return 0;
}
