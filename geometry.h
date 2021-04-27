#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include "helper_cuda.h"

class Particle {
private:
  float3 position;
  float3 velocity;

public:
  __host__ __device__ __forceinline__ void set_position(float3 pos) {
    position = pos;
  }

  __host__ __device__ __forceinline__ float3 get_position() {
    return position;
  }

  __host__ __device__ __forceinline__ void set_velocity(float3 vel) {
    velocity = vel;
  }

  __host__ __device__ __forceinline__ float3 get_velocity() {
    return velocity;
  }
};

class Particles {
private:
  thrust::host_vector<Particle> host_particle_list;
  thrust::device_vector<Particle> device_particle_list;

#if defined(__CUDA_ARCH__)
  thrust::device_vector<Particle>& get_vector() { return device_particle_list; }
#else
  thrust::host_vector<Particle>& get_vector() { return host_particle_list; }
#endif

public:
  __host__ __device__ Particles() = default;

  __host__ __device__ Particles(float *s_x, float *s_y, float *s_z, size_t ct) {
    for (size_t i = 0; i < ct; i++) {
      Particle p;
      p.set_position(make_float3(s_x[i], s_y[i], s_z[i]));
      get_vector().push_back(p);
    }
  }

  __host__ __device__ Particles(float *s_x, float *s_y, float *s_z, float *v_x,
                     float *v_y, float *v_z, size_t ct) {
    for (size_t i = 0; i < ct; i++) {
      Particle p;
      p.set_position(make_float3(s_x[i], s_y[i], s_z[i]));
      p.set_velocity(make_float3(v_x[i], v_y[i], v_z[i]));
      get_vector().push_back(p);
    }
  }

  Particles &operator=(const Particles &p) = default;

  __host__ __device__ __forceinline__ void add_particle(float s_x, float s_y, float s_z) {
    Particle p;
    p.set_position(make_float3(s_x, s_y, s_z));
    get_vector().push_back(p);
  }

  __host__ __device__ __forceinline__ void add_particle(float s_x, float s_y, float s_z,
                                                        float v_x, float v_y, float v_z) {
    Particle p;
    p.set_position(make_float3(s_x, s_y, s_z));
    p.set_velocity(make_float3(v_x, v_y, v_z));
    get_vector().push_back(p);
  }

  __host__  __device__ __forceinline__ Particle &get(size_t index) {
    Particle& p = &get_vector()[index];
    return p;
  }

  __host__  __device__ __forceinline__ size_t get_count() {
    return get_vector().size();
  }

  __host__  __device__ __forceinline__ float3 get_location(size_t index) {
    Particle& p = get_vector()[index];
    return p.get_position();
  }

  __host__  __device__ __forceinline__ void set_location(size_t index, float3 value) {
    Particle& p = get_vector()[index];
    return p.set_position(value);
  }

  void syncToHost() {
    host_particle_list = device_particle_list;
  }

  void syncToDevice() {
    device_particle_list = host_particle_list;
  }
};

class Bounding_box {
private:
  float3 m_p_min;
  float3 m_p_max;

public:
  __host__ __device__ Bounding_box() {
    m_p_min = make_float3(0.0f, 0.0f, 0.0f);
    m_p_max = make_float3(1.0f, 1.0f, 1.0f);
  }

  __host__ __device__ void compute_center(float3 &center) const {
    center.x = 0.5f * (m_p_min.x + m_p_max.x);
    center.y = 0.5f * (m_p_min.y + m_p_max.y);
    center.z = 0.5f * (m_p_min.z + m_p_max.z);
  }

  __host__ __device__ __forceinline__ const float3 &get_max() const {
    return m_p_max;
  }

  __host__ __device__ __forceinline__ const float3 &get_min() const {
    return m_p_min;
  }

  __host__ __device__ bool contains(const float3 &p) const {
    return p.x >= m_p_min.x && p.y < m_p_max.x && p.y >= m_p_min.y &&
           p.y < m_p_max.y && p.z < m_p_max.z && p.z >= m_p_min.z;
  }

  __host__ __device__ void set(float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z) {
    m_p_min.x = min_x;
    m_p_min.y = min_y;
    m_p_min.z = min_z;
    m_p_max.x = max_x;
    m_p_max.y = max_y;
    m_p_max.z = max_z;
  }
};
