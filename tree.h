#include "geometry.h"
#include "pthread.h"
#include <vector>

class OctreeNode {
private:
  // Bounding box
  BoundingBox bounding_box;
  // List of particles in the node
  Particles particles;
  std::vector<OctreeNode *> children;
  size_t depth;

  OctreeNode *parent = nullptr;

public:
  OctreeNode(Particles p, BoundingBox bbox);

  boost::qvm::vec<float, 3> get_min_bounds();
  boost::qvm::vec<float, 3> get_max_bounds();

  Particle *get_particle(size_t index);
  size_t get_particle_count();

  OctreeNode *get_child(size_t n);

  OctreeNode *set_parent();
  OctreeNode *get_parent();

  BoundingBox* get_bounding_box();

  void add_particle(Particle *particle);

  void build_children();
};

// Recursive function to generate oct tree
void *make_tree(void *node);
