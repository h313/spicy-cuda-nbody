#include "tree.h"

#include "pthread.h"
#include <boost/qvm/vec_access.hpp>
#include <iostream>
#include <omp.h>

using namespace boost::qvm;

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

BoundingBox &OctreeNode::get_bounding_box() { return bounding_box; }

void OctreeNode::add_particle(Particle particle) {
  particles.add_particle(particle);
}

Particle *OctreeNode::get_particle(size_t index) {
  return &particles.get(index);
}

size_t OctreeNode::get_particle_count() { return particles.get_count(); }

OctreeNode *OctreeNode::get_child(size_t n) { return children[n]; }

void OctreeNode::set_parent(OctreeNode *p) { this->parent = p; }

OctreeNode *OctreeNode::get_parent() { return this->parent; }

void OctreeNode::build_children() {
  // Split the BoundingBox into octants
  vec<float, 3> p_min = bounding_box.get_min();
  vec<float, 3> p_max = bounding_box.get_max();
  vec<float, 3> center = bounding_box.get_center();

  // Create OctreeNodes for each octant and add them to the children
  Particles empty_particles;
  // Top-Left Front
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(p_min), A<1>(center), A<2>(center),
                                   A<0>(center), A<2>(p_max), A<2>(p_max))));

  // Top-Right Front
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(center), A<1>(center), A<2>(center),
                                   A<0>(p_max), A<1>(p_max), A<2>(p_max))));

  // Bottom-Left Front
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(p_min), A<1>(p_min), A<2>(center),
                                   A<0>(center), A<1>(center), A<2>(p_max))));

  // Bottom-Right Front
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(center), A<1>(p_min), A<2>(center),
                                   A<0>(p_max), A<1>(center), A<2>(p_max))));

  // Top-Left Back
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(p_min), A<1>(center), A<2>(p_min),
                                   A<0>(center), A<1>(p_max), A<2>(center))));

  // Top-Right Back
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(center), A<1>(center), A<2>(p_min),
                                   A<0>(p_max), A<1>(p_max), A<2>(center))));

  // Bottom-Left Back
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(p_min), A<1>(p_min), A<2>(p_min),
                                   A<0>(center), A<1>(center), A<2>(center))));

  // Bottom-Right Back
  children.push_back(new OctreeNode(
      empty_particles, BoundingBox(A<0>(center), A<1>(p_min), A<2>(p_min),
                                   A<0>(p_max), A<1>(center), A<2>(center))));

  // Assign parent node as this for each child
  for (int i = 0; i < 8; i++) {
    children[i]->set_parent(this);

    // Loop through all the particles and assign them to the
    // correct OctreeNode
    for (int it = 0; it < this->get_particle_count(); it++) {
      if (children[i]->get_bounding_box().contains(
              particles.get(it).get_position()))
        children[i]->add_particle(particles.get(it));
    }
  }
}

// Recursive function to generate oct tree called using
void *make_tree_unbounded(void *node) {
  pthread_t child_threads[8];
  bool pthread_active[8] = {false};
  // Create the children of the root vector
  OctreeNode *root = static_cast<OctreeNode *>(node);
  if (root->get_particle_count() > 1) {
    root->build_children();

    // Now build chidren for each of that root's children as well
    for (size_t i = 0; i < 8; i++) {
      OctreeNode *child = root->get_child(i);
      // Only spawn a new thread if we have more than one particle in the area
      if (child->get_particle_count() > 1) {
        pthread_create(&child_threads[i], NULL, make_tree_unbounded, (void *)child);
        pthread_active[i] = true;
      }
    }

    // Wait for the children to finish
    for (int i = 0; i < 8; i++) {
      if (pthread_active[i])
        pthread_join(child_threads[i], NULL);
    }
  }
}

// Initialize the semaphore
void init_semaphore(int max_threads) {
  sem_init(&thread_availible, 0, max_threads);
}

// Destroy the semaphore
void deinit_semaphore() {
  sem_destroy(&thread_availible);
}

// Recursive function to generate oct tree called using 
void *make_tree_bounded(void *node) {
  pthread_t child_threads[8];
  bool pthread_active[8] = {false};
  // Create the children of the root vector
  OctreeNode *root = static_cast<OctreeNode *>(node);
  if (root->get_particle_count() > 1) {
    root->build_children();

    // Now build children for each of that root's children as well
    for (size_t i = 0; i < 8; i++) {
      OctreeNode *child = root->get_child(i);
      // Only spawn a new thread if we have more than one particle in the area
      if (child->get_particle_count() > 1) {
        // When this passes fewer than 8 threads are running
        sem_wait(&thread_availible);
        pthread_create(&child_threads[i], NULL, make_tree_bounded, (void *)child);
        pthread_active[i] = true;
      }
    }

    // Wait for the children to finish
    for (int i = 0; i < 8; i++) {
      if(pthread_active[i])
        pthread_join(child_threads[i], NULL);
    }
  }
  // Increment the semaphore since this thread is done
  sem_post(&thread_availible);
}

// Recursive function to serially generate oct tree called using
void make_tree_serial(OctreeNode *node) {
  // Create the children of the root vector
  OctreeNode *root = node;
  if(root->get_particle_count() > 1) {
    root->build_children();

    // Now build children for each of that root's children as well
    for(size_t i = 0; i < 8; i++) {
      OctreeNode *child = root->get_child(i);
      // Only recurse down if we have more than one particle in the area
      if(child->get_particle_count() > 1) {
        make_tree_serial(child);
      }
    }
  }
}

// Recursive function to generate oct tree called using
void make_tree_openmp(OctreeNode *root) {
  // Create the children of the root vector
  omp_set_num_threads(8);
  if (root->get_particle_count() > 1) {
    root->build_children();

    // Now build chidren for each of that root's children as well
    #pragma omp parallel for
    for (size_t i = 0; i < 8; i++) {
      OctreeNode *child = root->get_child(i);
      // Only spawn a new thread if we have more than one particle in the area
      if (child->get_particle_count() > 1)
        make_tree_openmp(child);
    }
  }
}
