#include "tree.h"

OctreeNode::OctreeNode(BoundingBox bbox) : bounding_box(bbox) {
  children.reserve(8);
}

boost::qvm::vec<float, 3> OctreeNode::get_min_bounds() { return bounding_box.get_min(); }

boost::qvm::vec<float, 3> OctreeNode::get_max_bounds() { return bounding_box.get_max(); }

Particle *get_particle(size_t index) { return particles[index]; }

size_t get_particle_count() { return particles.size(); }

OctreeNode* get_child(size_t n) { return children[n]; }

BoundingBox* get_bounding_box() { return bounding_box; }

void add_particle(Particle *particle) { particles.push_back(particle); }

void OctreeNode::build_children() {
  // Split the BoundingBox into octants
  boost::qvm::vec p_min = bounding_box.get_min();
  boost::qvm::vec p_max = bounding_box.get_max();
  boost::qvm::vec center = bounding_box.get_center();


  // Create OctreeNodes for each octant and add them to the children vector
  // Top-Left Front
  children.push_back(OctreeNode(BoundingBox(A<0>(p_min),  A<1>(center), A<2>(center),
                                 A<0>(center), A<2>(p_max), A<2>(p_max))));

  // Top-Right Front
  children.push_back(OctreeNode(BoundingBox(A<0>(center), A<1>(center), A<2>(center),
                                 A<0>(p_max),   A<1>(p_max), A<2>(p_max))));

  // Bottom-Left Front
  children.push_back(OctreeNode(BoundingBox(A<0>(p_min),  A<1>(p_min), A<2>(center),
                                 A<0>(center), A<1>(center), A<2>(p_max))));

  // Bottom-Right Front
  children.push_back(OctreeNode(BoundingBox(A<0>(center),  A<1>(p_min), A<2>(center),
                                 A<0>(p_max), A<1>(center), A<2>(p_max))));

  // Top-Left Back
  children.push_back(OctreeNode(BoundingBox(A<0>(p_min),  A<1>(center), A<2>(p_min),
                                 A<0>(center), A<1>(p_max), A<2>(center))));

  // Top-Right Back
  children.push_back(OctreeNode(BoundingBox(A<0>(center),  A<1>(center), A<2>(p_min),
                                 A<0>(p_max), A<1>(p_max), A<2>(center))));

  // Bottom-Left Back
  children.push_back(OctreeNode(BoundingBox(A<0>(p_min),  A<1>(p_min), A<2>(p_min),
                                 A<0>(center), A<1>(center), A<2>(center))));  

  // Bottom-Right Back
  children.push_back(OctreeNode(BoundingBox(A<0>(center),  A<1>(p_min), A<2>(p_min),
                                 A<0>(p_max), A<1>(center), A<2>(center))));  


  // Assign parent node as this for each child
  for(int i = 0; i < 8; i++)
    children[i].set_parent(this)

  // Loop through all the particles and assign them to the correct OctreeNode
  for(int i = 0; i < particles.size(); i++) {
    for(int j = 0; j < 8; j++) {
      if(children[j].get_bounding_box().contains(particles[i]->get_position()))
        children[j].add_particle(particles[i]);
    }
  }
}

// Recursive function to generate oct tree called using
void make_tree(OctreeNode &root, BoundingBox &box, Particles &particles) {
  root.build_children();
}
