#include "generator.h"

#include <chrono>
#include <random>

Particles generate_random_particles(size_t count, float min, float max) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> distrib(min, max);

  Particles particles;
  for (int i = 0; i < count; i++) {
    float s_x = distrib(rng);
    float s_y = distrib(rng);
    float s_z = distrib(rng);
    float v_x = distrib(rng);
    float v_y = distrib(rng);
    float v_z = distrib(rng);
    particles.add_particle(s_x, s_y, s_z, v_x, v_y, v_z);
  }

  return particles;
}
