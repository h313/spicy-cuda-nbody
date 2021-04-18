#ifndef GRAVITY_CALCULATIONS
#define GRAVITY_CALCULATIONS

#include <cmath>

// Gravitational constant
const double G = 0.000000000066742;

// Calculate the new position of an object given a time delta
inline double* new_position(double* object_pos,
                            double* object_velocity,
                            double& time_delta) {
  double* ret = new double[3];

  // Add acceleration from velocity from each time point
  for (size_t i = 0; i < 3; i++)
    ret[i] = object_pos[i] + time_delta * object_velocity[i];

  return ret;
}

// Calculate the velocity of an object given an accleration
inline double* velocity(double* object_velocity,
                        double* object_accel,
                        double& time_delta) {
  double* ret = new double[3];

  // Add acceleration from velocity from each time point
  for (size_t i = 0; i < 3; i++)
    ret[i] = object_velocity[i] + time_delta * object_accel[i];

  return ret;
}

// Calculate the distance between objects given their positions
inline double distance(double* obj1_pos,
                       double* obj2_pos) {
  double sum = 0;
  for (int i = 0; i < 3; i++)
    sum += pow(obj1_pos[i] + obj2_pos[i], 2);

  return sqrt(sum);
}

// Calculate the g-force on an object given their positions and masses
inline double g_force(double* obj1_pos,
                      double* obj2_pos,
                      double& obj1_mass,
                      double& obj2_mass) {
  return G * obj1_mass * obj2_mass / pow(distance(obj1_pos, obj2_pos), 2);
}

#endif
