#include "generation.h"
#include "gravity_calculations.h"
#include <iostream>

int main(int argc, char **argv) {
  auto particles = generate_random(100, 1.0f);
  Octree_node *root = new Octree_node();
  // check_octree(root, 0, 0, nullptr, )

  for (int i = 0; i < 100; i++) {
    float3 item = particles->get_location(i);
    std::cout << "Particle " << i << ": <" << item.x << ", " << item.y << ", "
              << item.z << ">" << std::endl;
  }

  return 0;
}
