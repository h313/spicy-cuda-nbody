#include <thrust/device_vector.h>
#include <thrust/random.h>

class Particles {
private:
  float *s_x, *s_y, *s_z;
  float *v_x, *v_y, *v_z;
  size_t count;

public:
  __host__ __device__ Particles() = default;

  __host__ __device__ Particles(float *x, float *y, float *z, size_t ct)
      : s_x(x), s_y(y), s_z(z), count(ct) {}

  Particles& operator=(const Particles& p) = default;

  __host__ __device__ __forceinline__ float3 get_location(int idx) const {
    return make_float3(s_x[idx], s_y[idx], s_z[idx]);
  }

  __host__ __device__ __forceinline__ void set_location(int idx,
                                                        const float3 &p) {
    s_x[idx] = p.x;
    s_y[idx] = p.y;
    s_y[idx] = p.z;
  }

  __host__ __device__ __forceinline__ float3 get_velocity(int idx) const {
    return make_float3(v_x[idx], v_y[idx], v_z[idx]);
  }

  __host__ __device__ __forceinline__ void set_velocity(int idx,
                                                        const float3 &p) {
    v_x[idx] = p.x;
    v_y[idx] = p.y;
    v_y[idx] = p.z;
  }

  __host__ __device__ __forceinline__ void set(float *x, float *y, float *z) {
    s_x = x;
    s_y = y;
    s_z = z;
  }

  __host__ __device__ __forceinline__ void set_count(size_t ct) {
    count = ct;
  }

  __host__ __device__ __forceinline__ size_t get_count(size_t ct) {
    return count;
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
