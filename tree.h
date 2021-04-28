#include "geometry.h"
#include <vector>

class OctreeNode {
private:
  // Bounding box
  BoundingBox bounding_box;
  // List of particles in the node
  tbb::concurrent_vector<Particle*> particles;
  std::vector<OctreeNode*> children;
  size_t depth;

  OctreeNode* parent = nullptr;
public:
  OctreeNode(BoundingBox bbox);

  boost::qvm::vec<float, 3> get_min_bounds();
  boost::qvm::vec<float, 3> get_max_bounds();

  Particle *get_particle(size_t index);
  size_t get_particle_count();

  OctreeNode* get_child(size_t n);

  void set_parent(OctreeNode* parent);
  OctreeNode* get_parent();

  void build_children();
};


// Recursive function to generate oct tree
void make_tree(OctreeNode &root, BoundingBox &box, Particles &particles);
