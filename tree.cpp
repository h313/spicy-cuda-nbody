#include "tree.h"

#include "pthread.h"

OctreeNode::OctreeNode(Particles p, BoundingBox bbox)
    : particles(p), bounding_box(bbox) {
  children.reserve(8);
}

boost::qvm::vec<float, 3> OctreeNode::get_min_bounds() { return bounding_box.get_min(); }

boost::qvm::vec<float, 3> OctreeNode::get_max_bounds() { return bounding_box.get_max(); }

BoundingBox* OctreeNode::get_bounding_box() { return bounding_box; }

void OctreeNode::add_particle(Particle *particle) { particles.push_back(particle); }

Particle *OctreeNode::get_particle(size_t index) {
  return &particles.get(index);
}

size_t OctreeNode::get_particle_count() { return particles.get_count(); }

OctreeNode *OctreeNode::get_child(size_t n) { return children[n]; }

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
