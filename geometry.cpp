#include "geometry.h"
#include <boost/qvm/vec_access.hpp>

using namespace boost::qvm;
using namespace tbb;

void Particle::set_position(vec<float, 3> pos) { position = pos; }

vec<float, 3> Particle::get_position() { return position; }

void Particle::set_velocity(vec<float, 3> vel) { velocity = vel; }

vec<float, 3> Particle::get_velocity() { return velocity; }

Particles::Particles(float &s_x, float &s_y, float &s_z, size_t &ct) {
  for (size_t i = 0; i < ct; i++) {
    Particle p;
    p.set_position(vec<double, 3>{{s_x, s_y, s_z}});
    particle_list.push_back(p);
  }
}

Particles::Particles(float *s_x, float *s_y, float *s_z, float *v_x, float *v_y,
                     float *v_z, size_t &ct) {
  for (size_t i = 0; i < ct; i++) {
    Particle p;
    p.set_position(vec<double, 3>{{s_x[i], s_y[i], s_z[i]}});
    p.set_velocity(vec<double, 3>{{v_x[i], v_y[i], v_z[i]}});
    particle_list.push_back(p);
  }
}

void Particles::add_particle(float s_x, float s_y, float s_z) {
  Particle p;
  p.set_position(vec<double, 3>{{s_x, s_y, s_z}});
  particle_list.push_back(p);
}

void Particles::add_particle(float s_x, float s_y, float s_z, float v_x,
                             float v_y, float v_z) {
  Particle p;
  p.set_position(vec<double, 3>{{s_x, s_y, s_z}});
  p.set_velocity(vec<double, 3>{{v_x, v_y, v_z}});
  particle_list.push_back(p);
}

void Particles::add_particle(vec<float, 3> loc) {
  Particle p;
  p.set_position(loc);
  particle_list.push_back(p);
}

void Particles::add_particle(vec<float, 3> loc, vec<float, 3> vel) {
  Particle p;
  p.set_position(loc);
  p.set_velocity(vel);
  particle_list.push_back(p);
}

Particle &Particles::get(size_t index) { return particle_list[index]; }

size_t Particles::get_count() { return particle_list.size(); }

vec<float, 3> Particles::get_location(size_t index) {
  Particle &p = particle_list[index];
  return p.get_position();
}

BoundingBox::BoundingBox(float x_min, float y_min, float z_min, float x_max,
                         float y_max, float z_max) {
  min = vec<double, 3>{{x_min, y_min, z_min}};
  max = vec<double, 3>{{x_max, y_max, z_max}};
}

vec<float, 3> BoundingBox::get_min() { return min; }

vec<float, 3> BoundingBox::get_max() { return max; }

bool BoundingBox::contains(vec<float, 3> item) {
  if (A<0>(item) >= A<0>(min) && A<0>(item) < A<0>(max) &&
      A<1>(item) >= A<1>(min) && A<1>(item) < A<1>(max) &&
      A<2>(item) >= A<2>(min) && A<2>(item) < A<2>(max))
    return true;
  else
    return false;
}