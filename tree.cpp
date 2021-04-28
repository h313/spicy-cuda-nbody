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

void OctreeNode::build_children() {
  // Split the BoundingBox into octants
  // Create OctreeNodes for each octant and add them to the children vector
  // Assign parent node as this for each child
  // Loop through all the particles and assign them to the correct OctreeNode
}

// Recursive function to generate oct tree called using
void make_tree(OctreeNode &root, BoundingBox &box, Particles &particles) {
  root.build_children();
}
