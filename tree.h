#include "geometry.h"
#include "pthread.h"
#include <vector>
#include <semaphore.h>

// Create the semaphore to limit the number of threads
sem_t thread_availible;

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

  void set_parent(OctreeNode* p);
  OctreeNode *get_parent();

  BoundingBox &get_bounding_box();

  void add_particle(Particle particle);

  void build_children();
};


// Initialize the semaphore
void init_semaphore(int max_threads);

// Destroy the semaphore
void deinit_semaphore();

// Recursive function to generate oct tree using unlimited pthreads
void *make_tree_unbounded(void *node);

// Recursive function to generate oct tree using limited number of pthreads
void *make_tree_bounded(void* node);

// Recursive function to generate oct tree serially
void make_tree_serial(OctreeNode *root);

void make_tree_openmp(OctreeNode *root);
