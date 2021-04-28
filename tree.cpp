#include "tree.h"

OctreeNode::OctreeNode(BoundingBox bbox) : bounding_box(bbox) {
  children.reserve(8);
}

boost::qvm::vec<float, 3> OctreeNode::get_min_bounds() {
  return bounding_box.get_min();
}

boost::qvm::vec<float, 3> OctreeNode::get_max_bounds() {
  return bounding_box.get_max();
}

Particle *get_particle(size_t index) {
  return particles[index];
}

size_t get_particle_count() {
  return particles.size();
}

OctreeNode* get_child(size_t n) {
  return children[n];
}

// Recursive function to generate oct tree called using
void make_tree(OctreeNode &root, BoundingBox &box, Particles &particles) {
  // 
}
