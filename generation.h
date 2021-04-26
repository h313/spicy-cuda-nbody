#include <chrono>
#include <random>
#include <memory>
#include "octree.h"

__host__ Particles* generate_random(size_t num_particles, float size) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> distrib(-1.0 * size, size);

  float *s_x = (float*) malloc(sizeof(float) * num_particles);
  float *s_y = (float*) malloc(sizeof(float) * num_particles);
  float *s_z = (float*) malloc(sizeof(float) * num_particles);

  for (int i = 0; i < num_particles; i++) {
    s_x[i] = distrib(rng);
    s_y[i] = distrib(rng);
    s_z[i] = distrib(rng);
  }

  Particles* particles = new Particles(s_x, s_y, s_z, num_particles);
  return particles;
}
