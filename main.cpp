#include "gravity_calculations.h"
#include "tree.h"
#include <iostream>

int main(int argc, char **argv) {
  Particles p;
  p.add_particle(1, 1, 1);

  BoundingBox bb(0, 0, 0, 100, 100, 100);
  OctreeNode *parent = new OctreeNode(p, bb);

  make_tree(static_cast<void *>(parent));
  return 0;
}
