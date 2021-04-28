#include "tree.h"
#include "pthread.h"

OctreeNode::OctreeNode(Particles p, BoundingBox bbox)
    : particles(p), bounding_box(bbox) {
  children.reserve(8);
}

boost::qvm::vec<float, 3> OctreeNode::get_min_bounds() {
  return bounding_box.get_min();
}

boost::qvm::vec<float, 3> OctreeNode::get_max_bounds() {
  return bounding_box.get_max();
}

Particle *OctreeNode::get_particle(size_t index) {
  return &particles.get(index);
}

size_t OctreeNode::get_particle_count() { return particles.get_count(); }

OctreeNode *OctreeNode::get_child(size_t n) { return children[n]; }

void OctreeNode::build_children() {
  // Split the BoundingBox into octants
  // Create OctreeNodes for each octant and add them to the children vector
  // Assign parent node as this for each child
  // Loop through all the particles and assign them to the correct OctreeNode
}

// Recursive function to generate oct tree called using
void *make_tree(void *node) {
  pthread_t child_threads[8];
  bool pthread_active[8];
  // Create the children in the root vector
  OctreeNode *root = static_cast<OctreeNode *>(node);
  root->build_children();

  // Now build chidren for each of that root's children as well
  for (size_t i = 0; i < 8; i++) {
    OctreeNode *child = root->get_child(i);
    // Only spawn a new thread if we have more than one particle in the area
    if (child->get_particle_count() > 1) {
      pthread_create(&child_threads[i], NULL, make_tree, (void *)child);
      pthread_active[i] = true;
    }
  }

  // Wait for the children to finish
  for (int i = 0; i < 8; i++) {
    if (pthread_active[i])
      pthread_join(child_threads[i], NULL);
  }
}
